"""A retrieval cache bound to ONE index version (Phase 8).

When the index moves to a new version, the service creates a new retriever, so results
computed for an old version can never be returned for a new one.
"""

from __future__ import annotations

from collections import OrderedDict

from core.semantic.models import SearchResult


class VersionedRetriever:
    def __init__(self, engine, version_id: int, max_entries: int = 64) -> None:
        self._engine = engine
        self.version_id = version_id
        self._max = max_entries
        self._cache: OrderedDict[tuple, list[SearchResult]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.last: list[SearchResult] = []  # results of the most recent search

    def __len__(self) -> int:
        return len(self._cache)

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[SearchResult]:
        key = (mode, top_k, " ".join(query.split()))
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
        else:
            self.misses += 1
            self._cache[key] = list(self._engine.search(query, mode=mode, top_k=top_k))
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        self.last = list(self._cache[key])
        return list(self.last)

    def stored_locations(self) -> dict[str, tuple[str, int, int]]:
        return self._engine.stored_locations()

    def close(self) -> None:
        self._engine.close()