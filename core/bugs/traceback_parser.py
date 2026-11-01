"""Parse Python and pytest tracebacks into frames.

The text is only parsed. No path found in it is ever opened or executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 20_000
MAX_FRAMES = 60
MAX_MESSAGE_CHARS = 500

PY_FRAME = re.compile(
    r'^\s*File "(?P<file>[^"\n]+)", line (?P<line>\d+)(?:, in (?P<func>.+?))?\s*$'
)
PYTEST_FRAME = re.compile(
    r"^(?P<file>(?:[A-Za-z]:)?[^\s:][^:\n]*?\.py):(?P<line>\d+): (?P<rest>.*)$"
)
E_LINE = re.compile(r"^E\s{2,}(?P<text>.*)$")
EXC_LINE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s?(?P<msg>.*))?$")
MARKERS_ONLY = re.compile(r"^\s*[\^~\s]+$")
CHAIN_PREFIXES = (
    "During handling of the above exception",
    "The above exception was the direct cause",
)
MAX_LINE_NUMBER = 10_000_000


@dataclass(frozen=True)
class Frame:
    file: str  # exactly as written in the traceback
    line: int
    func: str | None = None
    code: str = ""  # the source line the traceback printed, when it printed one


@dataclass
class ParsedTraceback:
    frames: list[Frame] = field(default_factory=list)  # the final error; innermost call last
    earlier_frames: list[Frame] = field(default_factory=list)  # from chained exceptions
    exception_type: str = ""
    exception_message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_frames(self) -> bool:
        return bool(self.frames)


@dataclass
class _Block:
    frames: list[Frame] = field(default_factory=list)
    exc_type: str = ""
    exc_message: str = ""


def _code_after(lines: list[str], index: int) -> str:
    """The source line a traceback prints under a frame, if there is one."""
    if index + 1 >= len(lines):
        return ""
    following = lines[index + 1]
    if not following.startswith((" ", "\t")) or MARKERS_ONLY.match(following):
        return ""
    if PY_FRAME.match(following):
        return ""
    return following.strip()[:300]


def _frame_from(raw: str, lines: list[str], index: int) -> Frame | None:
    python = PY_FRAME.match(raw)
    if python:
        line = int(python.group("line"))
        if line > MAX_LINE_NUMBER:
            return None
        func = (python.group("func") or "").strip() or None
        return Frame(python.group("file").strip(), line, func, _code_after(lines, index))
    pytest_style = PYTEST_FRAME.match(raw)
    if pytest_style:
        line = int(pytest_style.group("line"))
        if line > MAX_LINE_NUMBER:
            return None
        rest = pytest_style.group("rest").strip()
        func = rest[3:].strip() if rest.startswith("in ") else None
        code = _code_after(lines, index) if func else ""
        return Frame(pytest_style.group("file").strip(), line, func, code)
    return None


def _set_exception(block: _Block, text: str) -> bool:
    match = EXC_LINE.match(text.strip())
    if match is None:
        return False
    block.exc_type = match.group("type")
    block.exc_message = (match.group("msg") or "")[:MAX_MESSAGE_CHARS]
    return True


def parse_traceback(text: str) -> ParsedTraceback:
    """Parse a traceback. Never raises: problems become warnings."""
    result = ParsedTraceback()
    if not isinstance(text, str) or not text.strip():
        result.warnings.append("The input is empty.")
        return result
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        result.warnings.append(f"Only the first {MAX_INPUT_CHARS:,} characters were read.")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = [_Block()]
    for index, raw in enumerate(lines):
        block = blocks[-1]
        stripped = raw.strip()
        if stripped.startswith("Traceback (most recent call last)"):
            if block.frames or block.exc_type:
                blocks.append(_Block())
            continue
        if stripped.startswith(CHAIN_PREFIXES):
            continue
        frame = _frame_from(raw, lines, index)
        if frame is not None:
            block.frames.append(frame)
            continue
        e_line = E_LINE.match(raw)
        if e_line:
            body = e_line.group("text")
            if not block.exc_type and not _set_exception(block, body):
                if body.startswith("assert "):
                    block.exc_type = "AssertionError"
                    block.exc_message = body[:MAX_MESSAGE_CHARS]
            continue
        if block.frames and raw and not raw[0].isspace() and not block.exc_type:
            _set_exception(block, raw)

    framed = [b for b in blocks if b.frames]
    if not framed:
        result.warnings.append("No 'File ..., line N' or 'path.py:N:' references were found.")
        found = next((b for b in reversed(blocks) if b.exc_type), None)
        if found is not None:
            result.exception_type, result.exception_message = found.exc_type, found.exc_message
        return result
    last = framed[-1]
    result.frames = last.frames
    result.earlier_frames = [f for b in framed[:-1] for f in b.frames]
    source = last if last.exc_type else next((b for b in reversed(framed) if b.exc_type), last)
    result.exception_type, result.exception_message = source.exc_type, source.exc_message
    if len(result.frames) > MAX_FRAMES:
        result.frames = result.frames[-MAX_FRAMES:]
        result.warnings.append(f"Only the last {MAX_FRAMES} frames were kept.")
    if not result.exception_type:
        result.warnings.append("No exception line was found after the frames.")
    return result