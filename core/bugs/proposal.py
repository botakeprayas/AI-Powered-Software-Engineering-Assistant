"""Turn model text into a reviewable fix and a test draft.

The diff is computed here with difflib, never written by the model. Nothing is applied to
the project: a proposal holds the patched text in memory until the sandbox verifies it.
"""

from __future__ import annotations

import ast
import difflib
import re
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path

from core.bugs.mapping import is_test_path, normalize_path

PROPOSED_LABEL = "PROPOSED, NOT VERIFIED"
DRAFT_LABEL = "DRAFT, NOT RUN"
FIX_PROPOSED = "proposed"
FIX_NONE = "none"
FIX_REJECTED = "rejected"
DRAFT_READY = "drafted"
DRAFT_NONE = "none"
DRAFT_REJECTED = "rejected"
MAX_TEST_CHARS = 4000

BLOCKED_MODULES = frozenset(
    {
        "ctypes", "subprocess", "socket", "ssl", "urllib", "http", "ftplib", "smtplib",
        "telnetlib", "requests", "httpx", "aiohttp", "multiprocessing", "shutil", "pty",
        "webbrowser", "importlib", "pickle", "marshal",
    }
)
BLOCKED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input"})
BLOCKED_ATTRS = frozenset(
    {
        "system", "popen", "startfile", "fork", "kill", "rmtree", "unlink", "rmdir",
        "removedirs", "__subclasses__", "__globals__", "__builtins__",
    }
)

NO_FIX = re.compile(r"NO_FIX\s*:\s*(?P<why>.*)", re.IGNORECASE)
NO_TEST = re.compile(r"NO_TEST\s*:\s*(?P<why>.*)", re.IGNORECASE)
RATIONALE = re.compile(r"^RATIONALE\s*:\s*(?P<text>.*)$", re.IGNORECASE | re.MULTILINE)
BLOCK = re.compile(
    r"^FILE\s*:\s*(?P<file>[^\n]+?)[ \t]*\n<{7} SEARCH[ \t]*\n(?P<search>.*?)\n={7}[ \t]*\n"
    r"(?P<replace>.*?)\n?>{7} REPLACE",
    re.MULTILINE | re.DOTALL,
)
TEST_BLOCK = re.compile(
    r"FILE\s*:\s*(?P<file>\S+)[ \t]*\n```(?:python|py)?[ \t]*\n(?P<code>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
CODE_ONLY = re.compile(r"```(?:python|py)?[ \t]*\n(?P<code>.*?)\n```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Edit:
    file: str
    search: str
    replace: str


@dataclass
class FixProposal:
    status: str
    rationale: str = ""
    edits: list[Edit] = field(default_factory=list)
    patched: dict[str, str] = field(default_factory=dict)  # file -> new full text (in memory)
    diff: str = ""
    changed_lines: int = 0
    problems: list[str] = field(default_factory=list)
    raw: str = ""
    label: str = PROPOSED_LABEL

    @property
    def usable(self) -> bool:
        return self.status == FIX_PROPOSED and bool(self.patched)


@dataclass
class RegressionDraft:
    status: str
    path: str = ""
    code: str = ""
    reason: str = ""
    problems: list[str] = field(default_factory=list)
    raw: str = ""
    label: str = DRAFT_LABEL

    @property
    def usable(self) -> bool:
        return self.status == DRAFT_READY and bool(self.code)


def _risk_hits(tree: ast.AST) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    hits.append((f"imports {root}", node.lineno))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in BLOCKED_MODULES:
                hits.append((f"imports {root}", node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                hits.append((f"calls {func.id}()", node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            hits.append((f"uses .{node.attr}", node.lineno))
    return hits


def find_risks(source: str) -> list[str]:
    """Risky constructs in Python source (network, processes, file removal, eval, ...)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    hits = sorted(_risk_hits(tree), key=lambda hit: hit[1])  # source order, not tree order
    return [f"{message} (line {line})" for message, line in hits]


def risk_counter(source: str) -> Counter[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return Counter()
    return Counter(message for message, _ in _risk_hits(tree))


def _strip_fences(text: str) -> str:
    return re.sub(r"(?m)^[ \t]*```[^\n]*\n?", "", text)


def _diff_for(file: str, old: str, new: str) -> tuple[str, int]:
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{file}", tofile=f"b/{file}",
        )
    )
    changed = sum(
        1 for line in lines
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    return text, changed


def build_fix_proposal(
    raw: str,
    read_file: Callable[[str], str | None],
    allowed_files: Collection[str],
    max_edits: int = 3,
    max_changed_lines: int = 40,
) -> FixProposal:
    """Parse SEARCH/REPLACE blocks, apply them in memory and compute the diff."""
    text = _strip_fences(raw.replace("\r\n", "\n"))
    found = RATIONALE.search(text)
    rationale = found.group("text").strip()[:300] if found else ""
    blocks = list(BLOCK.finditer(text))
    declined = NO_FIX.search(text)
    if declined and not blocks:
        return FixProposal(FIX_NONE, declined.group("why").strip()[:300], raw=raw)
    if not blocks:
        problem = "No SEARCH/REPLACE block was found in the model's answer."
        return FixProposal(FIX_REJECTED, rationale, problems=[problem], raw=raw)

    allowed = set(allowed_files)
    problems: list[str] = []
    if len(blocks) > max_edits:
        problems.append(f"The proposal has {len(blocks)} edits; the limit is {max_edits}.")
    original: dict[str, str] = {}
    current: dict[str, str] = {}
    edits: list[Edit] = []
    for block in blocks[:max_edits]:
        file = normalize_path(block.group("file"))
        search, replace = block.group("search"), block.group("replace")
        if file not in allowed or is_test_path(file):
            problems.append(f"{file} is not an editable file.")
            continue
        if file not in original:
            existing = read_file(file)
            if existing is None:
                problems.append(f"{file} could not be read.")
                continue
            original[file] = current[file] = existing
        if not search.strip():
            problems.append(f"{file}: the SEARCH text is empty.")
            continue
        if search == replace:
            problems.append(f"{file}: the edit changes nothing.")
            continue
        count = current[file].count(search)
        if count != 1:
            detail = "was not found" if count == 0 else f"matches {count} places"
            problems.append(f"{file}: the SEARCH text {detail}; it must match exactly once.")
            continue
        current[file] = current[file].replace(search, replace, 1)
        edits.append(Edit(file, search, replace))
    if problems:
        return FixProposal(FIX_REJECTED, rationale, edits, problems=problems, raw=raw)

    patched = {name: new for name, new in current.items() if new != original[name]}
    diffs: list[str] = []
    changed_total = 0
    for name in sorted(patched):
        old, new = original[name], patched[name]
        if name.endswith(".py"):
            try:
                ast.parse(new)
            except SyntaxError as exc:
                where = f"line {exc.lineno}"
                problems.append(f"{name}: the patched file has a syntax error at {where}.")
                continue
            added = risk_counter(new) - risk_counter(old)
            if added:
                problems.append(
                    f"{name}: the patch introduces risky code ({', '.join(sorted(added))})."
                )
                continue
        diff_text, changed = _diff_for(name, old, new)
        diffs.append(diff_text)
        changed_total += changed
    if not problems and changed_total > max_changed_lines:
        problems.append(
            f"The patch changes {changed_total} lines; the limit is {max_changed_lines}."
        )
    if problems:
        return FixProposal(FIX_REJECTED, rationale, edits, problems=problems, raw=raw)
    return FixProposal(
        FIX_PROPOSED, rationale, edits, patched, "".join(diffs), changed_total, raw=raw
    )


def _has_test_function(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test")
        for node in ast.walk(tree)
    )


def build_regression_draft(
    raw: str, fallback_slug: str, exists: Callable[[str], bool]
) -> RegressionDraft:
    """Check a generated test. The path is always a NEW file under tests/."""
    text = raw.replace("\r\n", "\n")
    block = TEST_BLOCK.search(text) or CODE_ONLY.search(text)
    declined = NO_TEST.search(text)
    if declined and block is None:
        return RegressionDraft(DRAFT_NONE, reason=declined.group("why").strip()[:300], raw=raw)
    if block is None:
        problem = "No fenced Python code block was found in the model's answer."
        return RegressionDraft(DRAFT_REJECTED, problems=[problem], raw=raw)
    code = block.group("code").strip("\n") + "\n"
    given = block.groupdict().get("file") or ""
    stem = Path(given.replace("\\", "/")).stem if given else ""
    stem = re.sub(r"^test_(?:regression_)?", "", stem)
    slug = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_")[:40] or fallback_slug
    path = f"tests/test_regression_{slug}.py"
    counter = 2
    while exists(path):
        path = f"tests/test_regression_{slug}_{counter}.py"
        counter += 1

    problems: list[str] = []
    if len(code) > MAX_TEST_CHARS:
        problems.append(f"The test is longer than {MAX_TEST_CHARS:,} characters.")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        problems.append(f"The test has a syntax error at line {exc.lineno}.")
    else:
        if not _has_test_function(tree):
            problems.append("The draft defines no function whose name starts with 'test'.")
        problems += [f"Risky code is not allowed in a generated test: {hit}"
                     for hit in find_risks(code)]
    if problems:
        return RegressionDraft(DRAFT_REJECTED, path, code, problems=problems, raw=raw)
    return RegressionDraft(DRAFT_READY, path, code, raw=raw)