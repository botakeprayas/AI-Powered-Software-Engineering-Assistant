"""SQLite memory store: indexed versions, the per-file manifest and a symbol cache.

Part 1 provides versions and the file manifest. The derived columns of the `files` table and
the `symbols` table are filled by the incremental indexer (part 2). When a file's content
hash changes, its derived columns are reset to NULL, so old parse results can never survive
next to new content.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.memory.models import FileState, VersionRecord
from core.semantic.config import index_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash  TEXT,
    dirty        INTEGER NOT NULL DEFAULT 0,
    tree_digest  TEXT NOT NULL,
    branch       TEXT,
    indexed_at   TEXT NOT NULL,
    file_count   INTEGER NOT NULL DEFAULT 0,
    symbol_count INTEGER NOT NULL DEFAULT 0,
    parent_id    INTEGER,
    note         TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS files (
    path            TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL,
    size            INTEGER NOT NULL,
    is_python       INTEGER NOT NULL DEFAULT 1,
    version_changed INTEGER NOT NULL DEFAULT 0,
    status          TEXT,
    issues_json     TEXT,
    interface_hash  TEXT,
    imports_json    TEXT,
    assigned_json   TEXT,
    ignored_json    TEXT,
    edges_json      TEXT,
    nodes_json      TEXT
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol_key TEXT PRIMARY KEY,
    file       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    qualname   TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    docstring  TEXT NOT NULL,
    code       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_symbols_file ON symbols(file);
"""

DERIVED_COLUMNS = (
    "status", "issues_json", "interface_hash", "imports_json", "assigned_json",
    "ignored_json", "edges_json", "nodes_json",
)


def _upsert_file_sql() -> str:
    resets = ",\n    ".join(
        f"{name} = CASE WHEN files.content_hash = excluded.content_hash "
        f"THEN files.{name} ELSE NULL END"
        for name in DERIVED_COLUMNS
    )
    return (
        "INSERT INTO files (path, content_hash, size, is_python, version_changed)\n"
        "VALUES (?, ?, ?, ?, ?)\n"
        "ON CONFLICT(path) DO UPDATE SET\n"
        "    content_hash = excluded.content_hash,\n"
        "    size = excluded.size,\n"
        "    is_python = excluded.is_python,\n"
        "    version_changed = CASE WHEN files.content_hash = excluded.content_hash\n"
        "        THEN files.version_changed ELSE excluded.version_changed END,\n"
        f"    {resets}"
    )


UPSERT_FILE = _upsert_file_sql()


def memory_db_path(repo: Path) -> Path:
    """One memory database per repository path, next to the Phase 2 index."""
    digest = hashlib.sha256(str(Path(repo).resolve()).lower().encode("utf-8")).hexdigest()
    return index_dir() / f"memory-{digest[:10]}.db"


def _version(row: sqlite3.Row) -> VersionRecord:
    return VersionRecord(
        id=row["id"],
        commit_hash=row["commit_hash"],
        dirty=bool(row["dirty"]),
        tree_digest=row["tree_digest"],
        branch=row["branch"],
        indexed_at=row["indexed_at"],
        file_count=row["file_count"],
        symbol_count=row["symbol_count"],
        parent_id=row["parent_id"],
        note=row["note"],
    )


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._depth = 0

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error. Nested use joins the outer transaction."""
        self._depth += 1
        try:
            yield self._conn
        except BaseException:
            if self._depth == 1:
                self._conn.rollback()
            raise
        else:
            if self._depth == 1:
                self._conn.commit()
        finally:
            self._depth -= 1

    # ---- meta ---------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ---- versions -----------------------------------------------------------------
    def add_version(
        self,
        commit_hash: str | None,
        dirty: bool,
        tree_digest: str,
        branch: str | None = None,
        file_count: int = 0,
        symbol_count: int = 0,
        parent_id: int | None = None,
        note: str = "",
    ) -> int:
        indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.transaction():
            cursor = self._conn.execute(
                "INSERT INTO versions (commit_hash, dirty, tree_digest, branch, indexed_at, "
                "file_count, symbol_count, parent_id, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (commit_hash, int(dirty), tree_digest, branch, indexed_at,
                 file_count, symbol_count, parent_id, note),
            )
        return int(cursor.lastrowid)

    def update_version(
        self, version_id: int, file_count: int | None = None,
        symbol_count: int | None = None, note: str | None = None,
    ) -> None:
        with self.transaction():
            if file_count is not None:
                self._conn.execute(
                    "UPDATE versions SET file_count = ? WHERE id = ?", (file_count, version_id)
                )
            if symbol_count is not None:
                self._conn.execute(
                    "UPDATE versions SET symbol_count = ? WHERE id = ?", (symbol_count, version_id)
                )
            if note is not None:
                self._conn.execute(
                    "UPDATE versions SET note = ? WHERE id = ?", (note, version_id)
                )

    def get_version(self, version_id: int) -> VersionRecord | None:
        row = self._conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
        return None if row is None else _version(row)

    def set_current_version(self, version_id: int) -> None:
        self.set_meta("current_version", str(version_id))

    def current_version(self) -> VersionRecord | None:
        raw = self.get_meta("current_version")
        if raw is None or not raw.isdigit():
            return None
        return self.get_version(int(raw))

    def list_versions(self, limit: int = 50) -> list[VersionRecord]:
        rows = self._conn.execute("SELECT * FROM versions ORDER BY id DESC LIMIT ?", (limit,))
        return [_version(row) for row in rows]

    # ---- the file manifest ---------------------------------------------------------
    def put_file_state(self, state: FileState, version_id: int) -> None:
        """Record a file's hash. A changed hash resets the file's derived data."""
        with self.transaction():
            self._conn.execute(
                UPSERT_FILE,
                (state.path, state.content_hash, state.size, int(state.is_python), version_id),
            )

    def manifest(self) -> dict[str, FileState]:
        rows = self._conn.execute(
            "SELECT path, content_hash, size, is_python FROM files ORDER BY path"
        )
        return {
            row["path"]: FileState(
                row["path"], row["content_hash"], row["size"], bool(row["is_python"])
            )
            for row in rows
        }

    def delete_files(self, paths: Iterable[str]) -> None:
        """Remove files from the manifest together with their cached symbols."""
        rows = [(path,) for path in paths]
        with self.transaction():
            self._conn.executemany("DELETE FROM files WHERE path = ?", rows)
            self._conn.executemany("DELETE FROM symbols WHERE file = ?", rows)