"""Persistent vector cache (SQLite) plus an in-memory FAISS index.

SQLite is the durable source of truth: one row per symbol with location, content hash and
vector. FAISS is rebuilt from those rows after every sync, so they can never disagree.
The FAISS vector id IS the SQLite row id, which maps a search hit back to its symbol.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from core.semantic.models import TEXT_FORMAT_VERSION, Symbol, SyncStats
from services.embedding_service import EmbeddingError, EmbeddingService

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_key   TEXT NOT NULL UNIQUE,
    file         TEXT NOT NULL,
    qualname     TEXT NOT NULL,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
"""

UPSERT = """
INSERT INTO symbols
    (symbol_key, file, qualname, name, kind, start_line, end_line,
     content_hash, model, dim, vector)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol_key) DO UPDATE SET
    file=excluded.file, qualname=excluded.qualname, name=excluded.name, kind=excluded.kind,
    start_line=excluded.start_line, end_line=excluded.end_line,
    content_hash=excluded.content_hash, model=excluded.model, dim=excluded.dim,
    vector=excluded.vector
"""


def content_hash(symbol: Symbol, model_name: str) -> str:
    """Hash of everything that determines the vector. Line numbers are deliberately excluded."""
    payload = f"v{TEXT_FORMAT_VERSION}\0{model_name}\0{symbol.embedding_text()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return faiss


class SemanticIndex:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._faiss = None
        self._id_to_key: dict[int, str] = {}

    def close(self) -> None:
        self._conn.close()

    def sync(self, symbols: Sequence[Symbol], embedder: EmbeddingService) -> SyncStats:
        """Make the index match `symbols`, embedding only what is new or changed."""
        started = time.perf_counter()
        keys = [s.key for s in symbols]
        if len(set(keys)) != len(keys):
            raise ValueError("Symbol keys must be unique.")
        model = embedder.model_name
        existing = {
            row["symbol_key"]: row["content_hash"]
            for row in self._conn.execute("SELECT symbol_key, content_hash FROM symbols")
        }

        reuse: list[Symbol] = []
        todo: list[tuple[Symbol, str]] = []
        for symbol in symbols:
            digest = content_hash(symbol, model)
            if existing.get(symbol.key) == digest:
                reuse.append(symbol)
            else:
                todo.append((symbol, digest))

        new_rows: list[tuple] = []
        if todo:
            # Embed BEFORE touching the database: if this fails, the old cache is untouched.
            vectors = embedder.embed([s.embedding_text() for s, _ in todo])
            if len(vectors) != len(todo):
                raise EmbeddingError("Embedder returned the wrong number of vectors.")
            for (s, digest), vec in zip(todo, vectors, strict=True):
                blob = np.ascontiguousarray(vec, dtype=np.float32).tobytes()
                new_rows.append(
                    (s.key, s.file, s.qualname, s.name, s.kind, s.start_line, s.end_line,
                     digest, model, int(vec.shape[0]), blob)
                )

        stale = sorted(set(existing) - set(keys))
        with self._conn:  # one transaction: commit on success, roll back on error
            self._conn.executemany(
                "UPDATE symbols SET file=?, start_line=?, end_line=? WHERE symbol_key=?",
                [(s.file, s.start_line, s.end_line, s.key) for s in reuse],
            )
            self._conn.executemany(UPSERT, new_rows)
            self._conn.executemany(
                "DELETE FROM symbols WHERE symbol_key=?", [(k,) for k in stale]
            )
        self._rebuild_faiss(embedder.dimension)

        stats = SyncStats(
            total=len(symbols),
            embedded=len(todo),
            reused=len(reuse),
            removed=len(stale),
            embedded_files=sorted({s.file for s, _ in todo}),
            seconds=time.perf_counter() - started,
        )
        logger.info("Index sync: %s", stats)
        return stats

    def _rebuild_faiss(self, dim: int) -> None:
        faiss = _import_faiss()
        rows = self._conn.execute(
            "SELECT id, symbol_key, vector FROM symbols WHERE dim = ? ORDER BY id", (dim,)
        ).fetchall()
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._id_to_key = {}
        if rows:
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
            index.add_with_ids(np.ascontiguousarray(matrix), ids)
            self._id_to_key = {r["id"]: r["symbol_key"] for r in rows}
        self._faiss = index

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return (symbol_key, cosine_similarity), best first."""
        if self._faiss is None:
            raise RuntimeError("Semantic index is not built yet; call sync() first.")
        if self._faiss.ntotal == 0:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self._faiss.search(query, min(k, self._faiss.ntotal))
        return [
            (self._id_to_key[int(i)], float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i != -1
        ]

    def stored_locations(self) -> dict[str, tuple[str, int, int]]:
        """symbol_key -> (file, start_line, end_line), as stored in SQLite."""
        rows = self._conn.execute("SELECT symbol_key, file, start_line, end_line FROM symbols")
        return {r["symbol_key"]: (r["file"], r["start_line"], r["end_line"]) for r in rows}