"""Parse a unified diff (git diff or diff -u) into files, hunks and change blocks.

The text is only parsed. No path found in it is opened here, and nothing is executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_DIFF_CHARS = 400_000
MAX_FILES = 200
MAX_HUNKS_PER_FILE = 500

ADDED = "added"
DELETED = "deleted"
MODIFIED = "modified"
RENAMED = "renamed"

HUNK = re.compile(
    r"^@@ -(?P<os>\d+)(?:,(?P<oc>\d+))? \+(?P<ns>\d+)(?:,(?P<nc>\d+))? @@(?P<heading>.*)$"
)
QUOTED_PAIR = re.compile(r'^"((?:[^"\\]|\\.)*)" "((?:[^"\\]|\\.)*)"$')


@dataclass(frozen=True)
class Block:
    """A run of changed lines inside a hunk, with no context line between them.

    `old_start` is the old line number of the first removed line. When nothing was removed
    it is the old line the new text goes BEFORE. `new_start` works the same way.
    """

    old_start: int
    old_lines: tuple[str, ...]
    new_start: int
    new_lines: tuple[str, ...]


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    heading: str = ""
    old_text: list[tuple[int, str]] = field(default_factory=list)  # context + removed lines
    new_text: list[tuple[int, str]] = field(default_factory=list)  # context + added lines
    blocks: list[Block] = field(default_factory=list)
    truncated: bool = False  # fewer lines than the header promised


@dataclass
class FileDiff:
    old_path: str | None = None
    new_path: str | None = None
    status: str = MODIFIED
    hunks: list[Hunk] = field(default_factory=list)
    similarity: int | None = None
    binary: bool = False

    @property
    def lines_added(self) -> int:
        return sum(len(b.new_lines) for h in self.hunks for b in h.blocks)

    @property
    def lines_removed(self) -> int:
        return sum(len(b.old_lines) for h in self.hunks for b in h.blocks)

    @property
    def display_path(self) -> str:
        return self.new_path or self.old_path or "(unknown)"


@dataclass
class ParsedDiff:
    files: list[FileDiff] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clean_path(raw: str, strip_prefix: bool = True) -> str | None:
    """A repository-relative path from a diff header, or None for /dev/null."""
    text = raw.split("\t", 1)[0].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if text == "/dev/null":
        return None
    if strip_prefix and text[:2] in ("a/", "b/"):
        text = text[2:]
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text or None


def _split_git_header(rest: str) -> tuple[str | None, str | None]:
    """Old and new path from the text after 'diff --git '."""
    text = rest.strip()
    quoted = QUOTED_PAIR.match(text)
    if quoted:
        return _clean_path(quoted.group(1)), _clean_path(quoted.group(2))
    starts = [m.start() for m in re.finditer(" b/", text)]
    if not starts:
        return None, None
    for pos in starts:  # paths with spaces: prefer the split where both sides name one file
        left, right = _clean_path(text[:pos]), _clean_path(text[pos + 1 :])
        if left == right:
            return left, right
    pos = starts[0]
    return _clean_path(text[:pos]), _clean_path(text[pos + 1 :])


def _read_hunk(lines: list[str], start: int) -> tuple[Hunk | None, int]:
    """Read one hunk. The header's line counts decide where it ends, so a removed line
    that looks like a file header ('--- text') is never mistaken for one."""
    match = HUNK.match(lines[start])
    if match is None:
        return None, start + 1
    old_start = int(match.group("os"))
    old_count = int(match.group("oc") or 1)
    new_start = int(match.group("ns"))
    new_count = int(match.group("nc") or 1)
    hunk = Hunk(old_start, old_count, new_start, new_count, match.group("heading").strip())
    # With a count of 0 the header names the line BEFORE the change, so numbering starts next.
    old_line = old_start + 1 if old_count == 0 else old_start
    new_line = new_start + 1 if new_count == 0 else new_start
    left_old, left_new = old_count, new_count
    removed: list[str] = []
    added: list[str] = []
    first_old = first_new = 0

    def flush() -> None:
        if removed or added:
            hunk.blocks.append(Block(first_old, tuple(removed), first_new, tuple(added)))
            removed.clear()
            added.clear()

    index = start + 1
    while index < len(lines) and (left_old > 0 or left_new > 0):
        raw = lines[index]
        if raw.startswith("\\"):  # "\ No newline at end of file"
            index += 1
            continue
        tag, body = (raw[:1], raw[1:]) if raw else (" ", "")  # blank context lost its space
        if tag == " ":
            if left_old <= 0 or left_new <= 0:
                break
            flush()
            hunk.old_text.append((old_line, body))
            hunk.new_text.append((new_line, body))
            old_line += 1
            new_line += 1
            left_old -= 1
            left_new -= 1
        elif tag == "-":
            if left_old <= 0:
                break
            if not removed and not added:
                first_old, first_new = old_line, new_line
            removed.append(body)
            hunk.old_text.append((old_line, body))
            old_line += 1
            left_old -= 1
        elif tag == "+":
            if left_new <= 0:
                break
            if not removed and not added:
                first_old, first_new = old_line, new_line
            added.append(body)
            hunk.new_text.append((new_line, body))
            new_line += 1
            left_new -= 1
        else:
            break
        index += 1
    flush()
    if left_old > 0 or left_new > 0:
        hunk.truncated = True
    while index < len(lines) and lines[index].startswith("\\"):
        index += 1
    return hunk, index


def _add_file(result: ParsedDiff, file_diff: FileDiff) -> None:
    if len(result.files) < MAX_FILES:
        result.files.append(file_diff)
    elif not any("files were read" in w for w in result.warnings):
        result.warnings.append(f"Only the first {MAX_FILES} files were read.")


def parse_diff(text: str) -> ParsedDiff:
    """Parse a diff. Never raises: problems become warnings."""
    result = ParsedDiff()
    if not isinstance(text, str) or not text.strip():
        result.warnings.append("The diff is empty.")
        return result
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS]
        result.warnings.append(f"Only the first {MAX_DIFF_CHARS:,} characters were read.")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    current: FileDiff | None = None
    headers_seen = False
    orphan_warned = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            current = FileDiff()
            current.old_path, current.new_path = _split_git_header(line[11:])
            headers_seen = False
            _add_file(result, current)
        elif (
            line.startswith("--- ")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("+++ ")
        ):
            if current is None or headers_seen:  # plain 'diff -u' has no 'diff --git' line
                current = FileDiff()
                _add_file(result, current)
            headers_seen = True
            old = _clean_path(line[4:])
            new = _clean_path(lines[index + 1][4:])
            current.old_path, current.new_path = old, new
            if old is None and new is not None:
                current.status = ADDED
            elif new is None and old is not None:
                current.status = DELETED
            index += 1  # the '+++' line is consumed too
        elif current is not None and line.startswith("new file mode"):
            current.status = ADDED
        elif current is not None and line.startswith("deleted file mode"):
            current.status = DELETED
        elif current is not None and line.startswith("rename from "):
            current.old_path = _clean_path(line[12:], strip_prefix=False)
            current.status = RENAMED
        elif current is not None and line.startswith("rename to "):
            current.new_path = _clean_path(line[10:], strip_prefix=False)
            current.status = RENAMED
        elif current is not None and line.startswith("similarity index "):
            digits = re.match(r"similarity index (\d+)", line)
            current.similarity = int(digits.group(1)) if digits else None
        elif current is not None and (
            line.startswith("Binary files ") or line.startswith("GIT binary patch")
        ):
            current.binary = True
        elif line.startswith("@@"):
            hunk, index = _read_hunk(lines, index)
            if hunk is None:
                continue
            if current is None:
                if not orphan_warned:
                    message = "A hunk appears before any file header and was ignored."
                    result.warnings.append(message)
                    orphan_warned = True
            elif len(current.hunks) < MAX_HUNKS_PER_FILE:
                current.hunks.append(hunk)
            continue
        index += 1

    for file_diff in result.files:
        for hunk in file_diff.hunks:
            if hunk.truncated:
                result.warnings.append(
                    f"{file_diff.display_path}: the hunk at line {hunk.old_start} is "
                    "incomplete (the diff may have been cut off or edited)."
                )
    if not result.files:
        result.warnings.append("No file changes were found. Paste the output of git diff.")
    return result