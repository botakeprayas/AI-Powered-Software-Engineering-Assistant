"""Map a parsed diff onto the indexed repository: which functions and classes change.

The diff is first checked against the indexed file text. A diff that does not match is
refused rather than mapped by line number, because wrong line numbers give wrong symbols.
Only files that appear in the index are ever opened, so a path such as .env in a diff is
listed but never read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.graph.models import NODE_FILE
from core.impact.diff_parser import ADDED, DELETED, RENAMED, FileDiff, Hunk, ParsedDiff
from core.repo.extractor import normalize_newlines
from core.semantic.models import Symbol

SIDE_OLD = "old"  # the index holds the version BEFORE the change (a proposed change)
SIDE_NEW = "new"  # the index already holds the change
BASE_AUTO = "auto"
MODULE_LEVEL = "module-level"

OUT_MAPPED = "mapped"
OUT_NEW = "new_file"
OUT_NOT_PYTHON = "not_python"
OUT_NOT_INDEXED = "not_indexed"
OUT_MISMATCH = "mismatch"
OUT_UNPARSED = "unparsed"
OUT_BINARY = "binary"
OUT_EMPTY = "no_line_changes"


@dataclass
class ChangedUnit:
    """A function, a class, or the module-level code of a file that the change touches."""

    key: str  # the symbol key, or the file path for module-level code
    file: str
    kind: str  # function, method, class, ... or "module-level"
    qualname: str
    start_line: int = 0  # in the indexed version
    end_line: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    touched: set[int] = field(default_factory=set)  # existing lines that are changed
    inserted_after: set[int] = field(default_factory=set)  # lines added after these lines
    whole: str = ""  # "deleted" or "renamed" when the whole file changes

    @property
    def is_module_level(self) -> bool:
        return self.kind == MODULE_LEVEL

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_removed

    def describe(self) -> str:
        parts = []
        if self.whole:
            parts.append(f"the whole file is {self.whole}")
        if self.touched:
            parts.append(f"lines {ranges_text(self.touched)} changed")
        if self.inserted_after:
            parts.append(f"new lines inserted after line {ranges_text(self.inserted_after)}")
        return "; ".join(parts) or "changed"


@dataclass
class FileChange:
    path: str  # the path in the indexed repository (what the graph and index use)
    status: str
    diff_path: str = ""
    side: str = ""  # old or new, when the diff was matched against the file
    outcome: str = ""
    detail: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    units: list[ChangedUnit] = field(default_factory=list)


@dataclass
class ChangeSet:
    files: list[FileChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def units(self) -> list[ChangedUnit]:
        seen: set[str] = set()
        found: list[ChangedUnit] = []
        for change in self.files:
            for unit in change.units:
                if unit.key not in seen:
                    seen.add(unit.key)
                    found.append(unit)
        return found

    @property
    def changed_paths(self) -> list[str]:
        return [change.path for change in self.files if change.units]


def ranges_text(numbers: Iterable[int]) -> str:
    """[18, 19, 22] -> '18-19, 22'."""
    spans: list[str] = []
    start: int | None = None
    previous = 0
    for number in sorted(set(numbers)):
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            spans.append(str(start) if previous == start else f"{start}-{previous}")
            start = previous = number
    if start is not None:
        spans.append(str(start) if previous == start else f"{start}-{previous}")
    return ", ".join(spans)


def real_line_count(lines: Sequence[str]) -> int:
    """Number of real lines (the empty piece after a final newline is not a line)."""
    return len(lines) - 1 if lines and lines[-1] == "" else len(lines)


def indent_of(lines: Sequence[str]) -> int:
    """Indentation of the first non-blank line, or 0."""
    for text in lines:
        if text.strip():
            return len(text) - len(text.lstrip(" \t"))
    return 0


def def_indent(symbol: Symbol) -> int:
    first = symbol.code.split("\n", 1)[0]
    return len(first) - len(first.lstrip(" \t"))


def innermost(symbols: Sequence[Symbol], line: int) -> Symbol | None:
    inside = [s for s in symbols if s.start_line <= line <= s.end_line]
    if not inside:
        return None
    return min(inside, key=lambda s: (s.end_line - s.start_line, -s.start_line))


def insertion_owner(
    symbols: Sequence[Symbol], lines: Sequence[str], after: int, text: Sequence[str]
) -> Symbol | None:
    """Which symbol owns lines inserted after line `after`, or None for module level.

    Blank lines before the insertion are skipped, because a diff can place an insertion
    before or after blank lines with the same result. Inside a symbol, the insertion belongs
    to the innermost one. At the end of a symbol, deeper indentation than its def line means
    the new code is inside it.
    """
    total = real_line_count(lines)
    point = min(max(after, 0), total)
    while point > 0 and not lines[point - 1].strip():
        point -= 1
    first = next((t for t in text if t.strip()), "")
    if first.lstrip().startswith("@"):  # a new decorator belongs to the function below it
        following = after + 1
        while following <= total and not lines[following - 1].strip():
            following += 1
        for symbol in symbols:
            if symbol.start_line == following:
                return symbol
    indent = indent_of(text)
    candidates: list[tuple[Symbol, bool]] = []
    for symbol in symbols:
        if symbol.start_line <= point < symbol.end_line:
            candidates.append((symbol, True))
        elif symbol.end_line == point:
            candidates.append((symbol, False))
    candidates.sort(key=lambda c: (c[0].end_line - c[0].start_line, -c[0].start_line))
    for symbol, inside in candidates:
        if inside or indent > def_indent(symbol):
            return symbol
    return None


def verify_hunks(
    hunks: Sequence[Hunk], lines: Sequence[str], side: str
) -> tuple[bool, str, int]:
    """Do the diff's context and changed lines match the file? Returns (ok, why, checked)."""
    total = real_line_count(lines)
    checked = 0
    for hunk in hunks:
        pairs = hunk.old_text if side == SIDE_OLD else hunk.new_text
        for number, text in pairs:
            if number < 1 or number > total:
                message = f"line {number} is beyond the end of the file ({total} lines)"
                return False, message, checked
            if lines[number - 1].rstrip() != text.rstrip():
                return False, f"line {number} differs from the diff", checked
            checked += 1
    return True, "", checked


def _disk_reader(root: Path, files: set[str]) -> Callable[[str], list[str] | None]:
    base = root.resolve()

    def read(rel: str) -> list[str] | None:
        if rel not in files:  # only indexed files are ever opened
            return None
        target = (base / rel).resolve()
        if base not in target.parents:
            return None
        try:
            text = target.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return None
        return normalize_newlines(text).split("\n")

    return read


class IndexView:
    """The indexed repository as the mapper needs it: symbols, files and file text."""

    def __init__(
        self,
        symbols: Iterable[Symbol],
        files: Mapping[str, bool],
        reader: Callable[[str], list[str] | None] | None = None,
        root: Path | None = None,
    ) -> None:
        """`files` maps every indexed .py path to whether it was parsed."""
        self.files = dict(files)
        self.by_file: dict[str, list[Symbol]] = {}
        for symbol in symbols:
            self.by_file.setdefault(symbol.file, []).append(symbol)
        for items in self.by_file.values():
            items.sort(key=lambda s: (s.start_line, -s.end_line))
        self._reader = reader or _disk_reader(Path(root or "."), set(self.files))

    @classmethod
    def from_context(cls, context) -> IndexView:
        """Build from a Phase 4 GraphContext."""
        files: dict[str, bool] = {}
        for node_id, data in context.graph.nx_graph.nodes(data=True):
            if data.get("kind") == NODE_FILE:
                files[node_id] = bool(data.get("parsed", True))
        return cls(context.symbols.values(), files, root=Path(context.root))

    def symbols_in(self, path: str) -> list[Symbol]:
        return self.by_file.get(path, [])

    def read_lines(self, path: str) -> list[str] | None:
        return self._reader(path)


def _done(change: FileChange, outcome: str, detail: str) -> FileChange:
    change.outcome = outcome
    change.detail = detail
    return change


def _attribute(
    file_diff: FileDiff, symbols: list[Symbol], lines: list[str], path: str, side: str
) -> list[ChangedUnit]:
    """Attribute every changed line to the function, class or module level that owns it."""
    units: dict[str, ChangedUnit] = {}

    def unit_for(symbol: Symbol | None) -> ChangedUnit:
        key = symbol.key if symbol else path
        if key not in units:
            if symbol is None:
                units[key] = ChangedUnit(path, path, MODULE_LEVEL, f"module {path}")
            else:
                units[key] = ChangedUnit(
                    symbol.key, path, symbol.kind, symbol.qualname,
                    symbol.start_line, symbol.end_line,
                )
        return units[key]

    for hunk in file_diff.hunks:
        for block in hunk.blocks:
            if side == SIDE_OLD:
                first, touched = block.old_start, len(block.old_lines)
                other, touched_is_removed = block.new_lines, True
            else:
                first, touched = block.new_start, len(block.new_lines)
                other, touched_is_removed = block.old_lines, False
            if touched:
                owners = []
                for offset in range(touched):
                    line = first + offset
                    owner = unit_for(innermost(symbols, line))
                    owner.touched.add(line)
                    if touched_is_removed:
                        owner.lines_removed += 1
                    else:
                        owner.lines_added += 1
                    owners.append(owner)
                target = owners[0]  # replacement lines belong to the first changed line's owner
            else:
                after = first - 1
                target = unit_for(insertion_owner(symbols, lines, after, other))
                target.inserted_after.add(after)
            if touched_is_removed:
                target.lines_added += len(other)
            else:
                target.lines_removed += len(other)
    return list(units.values())


def _map_hunks(
    file_diff: FileDiff, view: IndexView, change: FileChange, path: str, base: str
) -> FileChange:
    lines = view.read_lines(path)
    if lines is None:
        return _done(change, OUT_MISMATCH, "the indexed file could not be read")
    sides = {BASE_AUTO: (SIDE_OLD, SIDE_NEW), SIDE_OLD: (SIDE_OLD,), SIDE_NEW: (SIDE_NEW,)}[base]
    chosen: str | None = None
    vacuous: str | None = None
    failures: list[str] = []
    for side in sides:
        ok, why, checked = verify_hunks(file_diff.hunks, lines, side)
        if ok and checked > 0:
            chosen = side
            break
        if ok and vacuous is None:
            vacuous = side
        if not ok:
            failures.append(f"{side} version: {why}")
    note = ""
    if chosen is None and vacuous is not None:
        chosen = vacuous
        note = " The diff has no context lines, so it could not be checked against the file."
    if chosen is None:
        detail = "the diff does not match the indexed file (" + "; ".join(failures) + ")"
        return _done(change, OUT_MISMATCH, detail)
    change.side = chosen
    symbols = view.symbols_in(path)
    change.units = _attribute(file_diff, symbols, lines, path, chosen)
    if not view.files.get(path, True):
        detail = (
            "the file could not be parsed when it was indexed, so only file-level impact "
            "is known" + note
        )
        return _done(change, OUT_UNPARSED, detail)
    version = (
        "version before the change" if chosen == SIDE_OLD else "version that already has the change"
    )
    return _done(change, OUT_MAPPED, f"matched by line number against the {version}.{note}")


def _whole_file(
    file_diff: FileDiff, view: IndexView, change: FileChange, action: str
) -> FileChange:
    old, new = file_diff.old_path, file_diff.new_path
    target = old if old in view.files else (new if new in view.files else None)
    if target is None:
        return _done(
            change, OUT_NOT_INDEXED, f"the {action} file is not in the indexed repository"
        )
    if action == "deleted" and file_diff.hunks:
        lines = view.read_lines(target)
        if lines is None:
            return _done(change, OUT_MISMATCH, "the indexed file could not be read")
        ok, why, _ = verify_hunks(file_diff.hunks, lines, SIDE_OLD)
        if not ok:
            detail = f"the diff does not match the indexed file ({why})"
            return _done(change, OUT_MISMATCH, detail)
    change.path = target
    change.side = SIDE_OLD if target == old else SIDE_NEW
    for symbol in view.symbols_in(target):
        unit = ChangedUnit(
            symbol.key, target, symbol.kind, symbol.qualname,
            symbol.start_line, symbol.end_line, whole=action,
        )
        unit.lines_removed = symbol.end_line - symbol.start_line + 1
        change.units.append(unit)
    module = ChangedUnit(target, target, MODULE_LEVEL, f"module {target}", whole=action)
    module.lines_removed = change.lines_removed
    change.units.append(module)
    detail = f"file {action}: every importer of this path is affected"
    return _done(change, OUT_MAPPED, detail)


def _map_file(file_diff: FileDiff, view: IndexView, base: str) -> FileChange:
    shown = file_diff.display_path
    change = FileChange(
        path=shown, status=file_diff.status, diff_path=shown,
        lines_added=file_diff.lines_added, lines_removed=file_diff.lines_removed,
    )
    if file_diff.binary:
        return _done(change, OUT_BINARY, "binary file: not analyzed")
    if not shown.endswith(".py"):
        return _done(change, OUT_NOT_PYTHON, "not a Python file: not analyzed")
    if file_diff.status == RENAMED:
        return _whole_file(file_diff, view, change, "renamed")
    if file_diff.status == DELETED:
        return _whole_file(file_diff, view, change, "deleted")
    path = file_diff.new_path or ""
    if path not in view.files:
        if file_diff.status == ADDED:
            return _done(change, OUT_NEW, "new file: nothing in the index depends on it yet")
        detail = "not in the indexed repository (skipped by the scanner, or outside it)"
        return _done(change, OUT_NOT_INDEXED, detail)
    change.path = path
    if not file_diff.hunks:
        return _done(change, OUT_EMPTY, "the diff has no line changes (a mode change only)")
    return _map_hunks(file_diff, view, change, path, base)


def map_diff(parsed: ParsedDiff, view: IndexView, base: str = BASE_AUTO) -> ChangeSet:
    """Find the functions, classes and module-level code that a diff touches.

    `base` is "old" (the index holds the version before the change), "new" (the index
    already holds it), or "auto" (try old first, then new).
    """
    if base not in (BASE_AUTO, SIDE_OLD, SIDE_NEW):
        raise ValueError(f"base must be 'auto', 'old' or 'new', got {base!r}")
    result = ChangeSet(warnings=list(parsed.warnings))
    for file_diff in parsed.files:
        result.files.append(_map_file(file_diff, view, base))
    return result