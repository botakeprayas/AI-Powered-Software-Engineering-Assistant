"""AST-based symbol extractor. Parsing never executes the analyzed code."""

from __future__ import annotations

import ast
import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from core.repo.models import FileIssue, RepoFile
from core.semantic.models import Symbol

logger = logging.getLogger(__name__)

DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass
class FileExtraction:
    path: str
    status: str  # parsed | empty | syntax_error | encoding_error | error
    symbols: list[Symbol] = field(default_factory=list)
    issues: list[FileIssue] = field(default_factory=list)


def normalize_newlines(text: str) -> str:
    """Make every line ending '\\n' so our line splitting matches the parser's numbering."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _kind(node: ast.AST, in_class: bool) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    base = "method" if in_class else "function"
    return f"async_{base}" if isinstance(node, ast.AsyncFunctionDef) else base


def _child_blocks(node: ast.AST) -> Iterator[list[ast.stmt]]:
    """Statement lists nested inside if/try/with/for/while/match (not inside functions)."""
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(node, attr, None)
        if isinstance(block, list):
            yield block
    for handler in getattr(node, "handlers", []):
        yield handler.body
    for case in getattr(node, "cases", []):
        yield case.body


def _make_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    prefix: str,
    rel: str,
    lines: list[str],
    seen: dict[str, int],
    in_class: bool,
) -> Symbol:
    qualname = f"{prefix}{node.name}"
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])  # decorators count
    end = node.end_lineno or node.lineno
    base_key = f"{rel}::{qualname}"
    seen[base_key] = seen.get(base_key, 0) + 1
    key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"
    return Symbol(
        key=key,
        file=rel,
        kind=_kind(node, in_class),
        name=node.name,
        qualname=qualname,
        start_line=start,
        end_line=end,
        docstring=ast.get_docstring(node) or "",
        code="\n".join(lines[start - 1 : end]),
    )


def _collect(
    body: list[ast.stmt],
    prefix: str,
    rel: str,
    lines: list[str],
    out: list[Symbol],
    seen: dict[str, int],
    in_class: bool,
) -> None:
    for node in body:
        if isinstance(node, DEF_NODES):
            symbol = _make_symbol(node, prefix, rel, lines, seen, in_class)
            out.append(symbol)
            if isinstance(node, ast.ClassDef):
                _collect(node.body, f"{symbol.qualname}.", rel, lines, out, seen, True)
        else:
            for block in _child_blocks(node):
                _collect(block, prefix, rel, lines, out, seen, in_class)


def extract_symbols(rel_path: str, text: str) -> list[Symbol]:
    """Parse source text and return its symbols. Raises SyntaxError/ValueError on bad code."""
    text = normalize_newlines(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. SyntaxWarning for odd escape sequences
        tree = ast.parse(text, filename=rel_path)
    lines = text.split("\n")
    out: list[Symbol] = []
    _collect(tree.body, "", rel_path, lines, out, {}, in_class=False)
    return out


def _failed(result: FileExtraction, status: str, message: str) -> FileExtraction:
    result.status = status
    result.issues.append(FileIssue(result.path, message))
    logger.warning("%s: %s", result.path, message)
    return result


def extract_file(root: Path, repo_file: RepoFile) -> FileExtraction:
    """Read and parse one file. Never raises: problems become warnings."""
    result = FileExtraction(path=repo_file.path, status="parsed")
    try:
        data = (Path(root) / repo_file.path).read_bytes()
    except OSError as exc:
        return _failed(result, "error", f"could not read file: {exc.strerror or exc}")
    if not data.strip():
        return _failed(result, "empty", "empty file (no symbols)")
    if b"\x00" in data:
        return _failed(result, "error", "contains NUL bytes (binary content); skipped")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return _failed(
            result, "encoding_error", f"not valid UTF-8 (bad byte at offset {exc.start}); skipped"
        )
    try:
        result.symbols = extract_symbols(repo_file.path, text)
    except SyntaxError as exc:
        where = f" at line {exc.lineno}" if exc.lineno else ""
        return _failed(result, "syntax_error", f"syntax error{where}: {exc.msg}; skipped")
    except (ValueError, RecursionError) as exc:
        return _failed(result, "error", f"could not be parsed: {exc}; skipped")
    return result