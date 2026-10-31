"""A read-only view of the CURRENTLY indexed repository version.

Built from the Phase 2 index (symbol locations) plus the files on disk. Only files that
appear in the index are ever opened, so files the scanner skipped (.env, keys, ...) are
never read. Nothing is cached, so every check sees the files as they are right now.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.repo.extractor import normalize_newlines


class IndexSnapshot:
    def __init__(self, root: Path, locations: Mapping[str, tuple[str, int, int]]) -> None:
        self.root = Path(root).resolve()
        self._locations = dict(locations)
        self.files = frozenset(file for file, _, _ in self._locations.values())

    @classmethod
    def from_engine(cls, root: Path, engine) -> IndexSnapshot:
        """`engine` is a synced Phase 2 CodeSearchEngine."""
        return cls(root, engine.stored_locations())

    def indexed_location(self, symbol_key: str) -> tuple[str, int, int] | None:
        return self._locations.get(symbol_key)

    def lines(self, rel_path: str) -> list[str] | None:
        """Lines of an indexed file as Phase 1 saw them, or None (not indexed, unreadable)."""
        if rel_path not in self.files:
            return None
        target = (self.root / rel_path).resolve()
        if self.root not in target.parents:
            return None
        try:
            text = target.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return None
        return normalize_newlines(text).split("\n")