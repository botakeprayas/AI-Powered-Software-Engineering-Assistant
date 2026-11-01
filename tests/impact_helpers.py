"""Helpers for the Phase 7 tests."""

from __future__ import annotations

import difflib
from pathlib import Path

from core.impact.changes import IndexView
from core.repo.extractor import extract_symbols

IMPACT_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "impact_sample"

CART_LINES = [
    '"""Cart."""',  # 1
    "import os",  # 2
    "",  # 3
    "",  # 4
    "class Cart:",  # 5
    '    """A cart."""',  # 6
    "",  # 7
    "    def add(self, item):",  # 8
    '        """Add an item."""',  # 9
    "        self.items.append(item)",  # 10
    "",  # 11
    "    def total(self):",  # 12
    '        """Sum."""',  # 13
    "        return sum(self.items)",  # 14
    "",  # 15
    "",  # 16
    "def helper(x):",  # 17
    '    """Help."""',  # 18
    "    return x + 1",  # 19
    "",  # 20
    "",  # 21
    "LIMIT = 10",  # 22
]
# Symbols: Cart 5-14, Cart.add 8-10, Cart.total 12-14, helper 17-19.
CART = "\n".join(CART_LINES) + "\n"


def replace_line(text: str, number: int, new: str) -> str:
    lines = text.split("\n")
    lines[number - 1] = new
    return "\n".join(lines)


def insert_after(text: str, number: int, *new: str) -> str:
    lines = text.split("\n")
    lines[number:number] = list(new)
    return "\n".join(lines)


def _unified(old: str, new: str, from_name: str, to_name: str, context: int) -> str:
    out = []
    for item in difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        from_name,
        to_name,
        n=context,
    ):
        if not item.endswith("\n"):  # git marks a missing final newline like this
            item += "\n\\ No newline at end of file\n"
        out.append(item)
    return "".join(out)


def make_diff(path: str, old: str, new: str, context: int = 3) -> str:
    head = f"diff --git a/{path} b/{path}\n"
    return head + _unified(old, new, f"a/{path}", f"b/{path}", context)


def make_new_diff(path: str, new: str) -> str:
    head = f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
    return head + _unified("", new, "/dev/null", f"b/{path}", 3)


def make_delete_diff(path: str, old: str) -> str:
    head = f"diff --git a/{path} b/{path}\ndeleted file mode 100644\n"
    return head + _unified(old, "", f"a/{path}", "/dev/null", 3)


def make_view(texts: dict[str, str], unparsed: tuple[str, ...] = ()) -> IndexView:
    """An IndexView over in-memory files, built with the same extractor as Phase 1."""
    symbols = []
    files = {}
    for path, text in texts.items():
        files[path] = path not in unparsed
        if path not in unparsed:
            symbols.extend(extract_symbols(path, text))
    lines = {path: text.split("\n") for path, text in texts.items()}
    return IndexView(symbols, files, lines.get)