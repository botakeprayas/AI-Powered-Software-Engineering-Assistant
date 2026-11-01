"""Map traceback frames onto indexed symbols. Only the index is consulted; no file is opened.

A frame is mapped only when the evidence supports it. Library frames, unknown files,
ambiguous paths and stale lines are refused with a reason, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from core.bugs.traceback_parser import Frame
from core.semantic.models import Symbol

MAPPED = "mapped"
MODULE_LEVEL = "module_level"
AMBIGUOUS = "ambiguous"
EXTERNAL = "external"
UNINDEXED = "unindexed"
MISMATCH = "mismatch"

STDLIB_DIR = re.compile(r"/python\d[\d.]*/lib/")
EXTERNAL_MARKERS = ("site-packages", "dist-packages", "/lib/python", "/lib64/python")


@dataclass(frozen=True)
class MappedFrame:
    frame: Frame
    status: str
    file: str | None
    symbol: Symbol | None
    reason: str
    is_test: bool

    @property
    def location(self) -> str:
        return f"{self.file or self.frame.file}:{self.frame.line}"


def normalize_path(raw: str) -> str:
    path = raw.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def is_external_path(path: str) -> bool:
    """Standard library, installed packages and interpreter-internal pseudo files."""
    norm = normalize_path(path)
    lowered = norm.lower()
    return (
        norm.startswith("<")
        or any(marker in lowered for marker in EXTERNAL_MARKERS)
        or STDLIB_DIR.search(lowered) is not None
    )


def is_test_path(path: str) -> bool:
    parts = normalize_path(path).lower().split("/")
    name = parts[-1]
    return (
        any(part in ("tests", "test") for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _common_suffix(a: list[str], b: list[str]) -> int:
    count = 0
    while count < min(len(a), len(b)) and a[-1 - count].lower() == b[-1 - count].lower():
        count += 1
    return count


def _squash(text: str) -> str:
    return " ".join(text.split())


def _line_text(symbol: Symbol, line: int) -> str | None:
    lines = symbol.code.split("\n")
    index = line - symbol.start_line
    return lines[index] if 0 <= index < len(lines) else None


class FrameMapper:
    def __init__(
        self,
        symbols: Sequence[Symbol],
        indexed_files: Collection[str] | None = None,
        root_hint: str | None = None,
    ) -> None:
        """`root_hint` is the repository folder as the traceback's machine saw it, if known."""
        self._by_file: dict[str, list[Symbol]] = {}
        for symbol in symbols:
            self._by_file.setdefault(symbol.file, []).append(symbol)
        self._files = set(indexed_files or ()) | set(self._by_file)
        self._root = normalize_path(root_hint).rstrip("/") if root_hint else ""

    def map(self, frames: Sequence[Frame]) -> list[MappedFrame]:
        return [self.map_frame(frame) for frame in frames]

    def map_frame(self, frame: Frame) -> MappedFrame:
        norm = normalize_path(frame.file)
        test = is_test_path(norm)

        def result(status: str, reason: str, file=None, symbol=None) -> MappedFrame:
            return MappedFrame(frame, status, file, symbol, reason, test)

        if is_external_path(norm):
            return result(EXTERNAL, "library or standard-library code, not part of the repository")
        file, how = self._match_file(norm)
        if file is None:
            status = AMBIGUOUS if how.startswith("several") else UNINDEXED
            return result(status, how)
        symbol = self._symbol_at(file, frame.line)
        if symbol is None:
            reason = f"line {frame.line} of {file} is outside every function and class"
            return result(MODULE_LEVEL, reason, file)
        if frame.code:
            seen = _line_text(symbol, frame.line)
            if seen is not None and _squash(seen) != _squash(frame.code):
                reason = (
                    "the line in the indexed file differs from the traceback "
                    "(the code changed, or this is another version)"
                )
                return result(MISMATCH, reason, file)
        return result(MAPPED, how, file, symbol)

    def _match_file(self, norm: str) -> tuple[str | None, str]:
        rel = norm
        if self._root and norm.lower().startswith(self._root.lower() + "/"):
            rel = norm[len(self._root) + 1 :]
        if rel in self._files:
            return rel, "path matches the indexed file"
        parts = rel.split("/")
        best: list[str] = []
        best_len = 0
        for candidate in self._files:
            candidate_parts = candidate.split("/")
            count = _common_suffix(parts, candidate_parts)
            if count == 0 or count not in (len(parts), len(candidate_parts)):
                continue
            if count > best_len:
                best, best_len = [candidate], count
            elif count == best_len:
                best.append(candidate)
        if len(best) == 1:
            return best[0], "matched by the end of the path (the traceback path is outside)"
        if len(best) > 1:
            return None, "several indexed files match this path: " + ", ".join(sorted(best)[:3])
        return None, "the file is not in the indexed repository"

    def _symbol_at(self, file: str, line: int) -> Symbol | None:
        inside = [s for s in self._by_file.get(file, []) if s.start_line <= line <= s.end_line]
        if not inside:
            return None
        return min(inside, key=lambda s: (s.end_line - s.start_line, -s.start_line))


def pick_primary(mapped: Sequence[MappedFrame]) -> MappedFrame | None:
    """The innermost mapped frame in non-test code; else the innermost mapped frame."""
    usable = [m for m in reversed(mapped) if m.status == MAPPED and m.symbol is not None]
    for item in usable:
        if not item.is_test:
            return item
    return usable[0] if usable else None


def select_frames(
    mapped: Sequence[MappedFrame], limit: int = 3, include_test: bool = True
) -> list[MappedFrame]:
    """Distinct mapped symbols, innermost first: up to `limit` repo frames plus one test frame."""
    chosen: list[MappedFrame] = []
    seen: set[str] = set()
    for item in reversed(mapped):
        if item.status != MAPPED or item.symbol is None or item.is_test:
            continue
        if item.symbol.key in seen:
            continue
        seen.add(item.symbol.key)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    if include_test:
        for item in reversed(mapped):
            if item.status == MAPPED and item.is_test and item.symbol is not None:
                if item.symbol.key not in seen:
                    chosen.append(item)
                break
    return chosen