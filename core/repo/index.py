"""SQLite symbol index with exact (case-insensitive substring) keyword search.

In-memory by default. Pass a file path to persist it instead; nothing else changes.
Queries are always parameterized. Only fixed SQL fragments are ever concatenated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from core.repo.models import KeywordHit
from core.semantic.models import Symbol

SEARCH_FIELDS = ("name", "docstring", "code")

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_key  TEXT NOT NULL UNIQUE,
    file        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    qualname    TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    docstring   TEXT NOT NULL,
    code        TEXT NOT NULL,
    name_lc     TEXT NOT NULL,
    qualname_lc TEXT NOT NULL,
    doc_lc      TEXT NOT NULL,
    code_lc     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
"""

INSERT = """
INSERT INTO symbols
    (symbol_key, file, kind, name, qualname, start_line, end_line, docstring, code,
     name_lc, qualname_lc, doc_lc, code_lc)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _row_to_symbol(row: sqlite3.Row) -> Symbol:
    return Symbol(
        key=row["symbol_key"],
        file=row["file"],
        kind=row["kind"],
        name=row["name"],
        qualname=row["qualname"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        docstring=row["docstring"],
        code=row["code"],
    )


def _score(row: sqlite3.Row, terms: list[str], query: str, chosen: tuple[str, ...]):
    score = 0
    matched: set[str] = set()
    if "name" in chosen and query in (row["name_lc"], row["qualname_lc"]):
        score += 100  # the whole query IS a symbol name
        matched.add("name")
    for term in terms:
        if "name" in chosen and (term in row["name_lc"] or term in row["qualname_lc"]):
            score += 40
            matched.add("name")
        if "docstring" in chosen and term in row["doc_lc"]:
            score += 15
            matched.add("docstring")
        if "code" in chosen and term in row["code_lc"]:
            score += 5
            matched.add("code")
    return score, tuple(f for f in SEARCH_FIELDS if f in matched)


class SymbolIndex:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        # check_same_thread=False: Streamlit reruns scripts on different threads.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def replace_all(self, symbols: Sequence[Symbol]) -> None:
        """Make the index contain exactly these symbols (one transaction)."""
        rows = [
            (s.key, s.file, s.kind, s.name, s.qualname, s.start_line, s.end_line, s.docstring,
             s.code, s.name.lower(), s.qualname.lower(), s.docstring.lower(), s.code.lower())
            for s in symbols
        ]
        with self._conn:
            self._conn.execute("DELETE FROM symbols")
            self._conn.executemany(INSERT, rows)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])

    def counts_by_kind(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT kind, COUNT(*) AS n FROM symbols GROUP BY kind")
        return {row["kind"]: row["n"] for row in rows}

    def search(
        self, query: str, fields: Iterable[str] = SEARCH_FIELDS, limit: int = 50
    ) -> list[KeywordHit]:
        """Every word of `query` must appear (case-insensitive) in at least one chosen field."""
        terms = query.lower().split()
        if not terms:
            raise ValueError("Query must not be empty.")
        requested = tuple(fields)
        chosen = tuple(f for f in SEARCH_FIELDS if f in requested)
        if not chosen or set(requested) - set(SEARCH_FIELDS):
            raise ValueError(f"fields must be a non-empty subset of {SEARCH_FIELDS}")
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        clauses: list[str] = []
        params: list[str] = []
        for term in terms:
            options: list[str] = []
            if "name" in chosen:
                options += ["instr(name_lc, ?) > 0", "instr(qualname_lc, ?) > 0"]
                params += [term, term]
            if "docstring" in chosen:
                options.append("instr(doc_lc, ?) > 0")
                params.append(term)
            if "code" in chosen:
                options.append("instr(code_lc, ?) > 0")
                params.append(term)
            clauses.append("(" + " OR ".join(options) + ")")
        sql = "SELECT * FROM symbols WHERE " + " AND ".join(clauses)

        whole = query.strip().lower()
        hits: list[KeywordHit] = []
        for row in self._conn.execute(sql, params).fetchall():
            score, matched = _score(row, terms, whole, chosen)
            hits.append(KeywordHit(_row_to_symbol(row), score, matched))
        hits.sort(key=lambda h: (-h.score, h.symbol.file, h.symbol.start_line))
        return hits[:limit]