"""Where symbols (functions, classes) come from.

FALLBACK: AstSymbolSource parses .py files directly with Python's ast module. It exists
because Phase 1's symbol data was not visible when Phase 2 was written. Once the Phase 1
API is known, replace it with a small adapter that implements SymbolSource.

Parsing never executes the analyzed code. Known limits: Python only; nested functions and
definitions inside if/try blocks are not indexed.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Protocol

from core.semantic.models import Symbol

logger = logging.getLogger(__name__)

SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "env", "__pycache__", "node_modules",
     ".pytest_cache", ".ruff_cache", ".idea", ".vscode"}
)


class SymbolSource(Protocol):
    def load(self) -> list[Symbol]:
        """Return every symbol currently in the repository."""
        ...


class AstSymbolSource:
    """FALLBACK symbol source based on Python's ast module."""

    def __init__(self, root: Path, max_file_bytes: int = 1_000_000) -> None:
        self.root = Path(root).resolve()
        self.max_file_bytes = max_file_bytes
        self.errors: list[str] = []  # files skipped, with the reason

    def load(self) -> list[Symbol]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Repository folder not found: {self.root}")
        self.errors = []
        symbols: list[Symbol] = []
        for path in self._python_files():
            symbols.extend(self._parse_file(path))
        return symbols

    def _python_files(self) -> list[Path]:
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            found.extend(Path(dirpath) / n for n in sorted(filenames) if n.endswith(".py"))
        return found

    def _skip(self, rel: str, reason: str) -> None:
        logger.warning("Skipping %s: %s", rel, reason)
        self.errors.append(f"{rel}: {reason}")

    def _parse_file(self, path: Path) -> list[Symbol]:
        rel = path.relative_to(self.root).as_posix()
        try:
            if path.stat().st_size > self.max_file_bytes:
                self._skip(rel, "file too large")
                return []
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(text, filename=rel)
        except (OSError, SyntaxError, ValueError) as exc:
            self._skip(rel, str(exc))
            return []
        lines = text.split("\n")  # matches the line numbering used by ast
        out: list[Symbol] = []
        self._collect(tree.body, "", rel, lines, out, {}, in_class=False)
        return out

    def _collect(self, body, prefix, rel, lines, out, seen, in_class) -> None:
        for node in body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            is_class = isinstance(node, ast.ClassDef)
            kind = "class" if is_class else ("method" if in_class else "function")
            qualname = f"{prefix}{node.name}"
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            end = node.end_lineno or node.lineno
            base_key = f"{rel}::{qualname}"
            seen[base_key] = seen.get(base_key, 0) + 1
            key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"
            out.append(
                Symbol(
                    key=key,
                    file=rel,
                    kind=kind,
                    name=node.name,
                    qualname=qualname,
                    start_line=start,
                    end_line=end,
                    docstring=ast.get_docstring(node) or "",
                    code="\n".join(lines[start - 1 : end]),
                )
            )
            if is_class:
                self._collect(node.body, f"{qualname}.", rel, lines, out, seen, in_class=True)