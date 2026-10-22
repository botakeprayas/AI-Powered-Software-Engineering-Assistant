"""CodeSearchEngine: keyword, semantic and hybrid (RRF) search over a repository."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from core.semantic.fusion import reciprocal_rank_fusion
from core.semantic.keyword import KeywordIndex
from core.semantic.models import SearchResult, Symbol, SyncStats
from core.semantic.store import SemanticIndex
from core.semantic.symbol_source import SymbolSource
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

MODES = ("keyword", "semantic", "hybrid")
KeywordFn = Callable[[str, int], list[tuple[str, float]]]


class CodeSearchEngine:
    def __init__(
        self,
        source: SymbolSource,
        embedder: EmbeddingService,
        db_path: Path,
        keyword_search: KeywordFn | None = None,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> None:
        """`keyword_search(query, limit)` may replace the built-in BM25 (e.g. Phase 1 search).

        It must return [(symbol_key, score)] best first, using this engine's symbol keys.
        """
        self._source = source
        self._embedder = embedder
        self._index = SemanticIndex(db_path)
        self._custom_keyword = keyword_search
        self._weights = (semantic_weight, keyword_weight)
        self._symbols: dict[str, Symbol] = {}
        self._keyword: KeywordIndex | None = None
        self._semantic_ready = False

    def close(self) -> None:
        self._index.close()

    def stored_locations(self) -> dict[str, tuple[str, int, int]]:
        return self._index.stored_locations()

    def sync(self, semantic: bool = True) -> SyncStats:
        """Load symbols and refresh the indexes. semantic=False skips embedding entirely.

        The keyword index is ready before embedding starts, so keyword search still works if
        the embedding model fails (the EmbeddingError is then raised to the caller).
        """
        symbols = self._source.load()
        self._symbols = {s.key: s for s in symbols}
        self._keyword = KeywordIndex(symbols)
        self._semantic_ready = False
        if not semantic:
            return SyncStats(total=len(symbols))
        stats = self._index.sync(symbols, self._embedder)
        self._semantic_ready = True
        return stats

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self._keyword is None:
            raise RuntimeError("Call sync() before search().")

        pool = max(top_k * 4, 20)  # candidates taken from each method before fusing
        semantic = self._semantic_ranked(query, pool) if mode != "keyword" else []
        keyword = self._keyword_ranked(query, pool) if mode != "semantic" else []
        sem_info = {key: (rank, sc) for rank, (key, sc) in enumerate(semantic, start=1)}
        kw_info = {key: (rank, sc) for rank, (key, sc) in enumerate(keyword, start=1)}

        if mode == "semantic":
            ordered = semantic
        elif mode == "keyword":
            ordered = keyword
        else:
            ordered = reciprocal_rank_fusion(
                [[k for k, _ in semantic], [k for k, _ in keyword]],
                weights=self._weights,
            )

        results: list[SearchResult] = []
        for key, score in ordered:
            if key not in self._symbols:
                continue
            s_rank, s_score = sem_info.get(key, (None, None))
            k_rank, k_score = kw_info.get(key, (None, None))
            results.append(
                SearchResult(self._symbols[key], score, s_rank, s_score, k_rank, k_score)
            )
            if len(results) == top_k:
                break
        return results

    def _semantic_ranked(self, query: str, pool: int) -> list[tuple[str, float]]:
        if not self._semantic_ready:
            raise RuntimeError(
                "The semantic index is not ready. Call sync() (without semantic=False) "
                "and make sure the embedding model loaded."
            )
        vector = self._embedder.embed([query])[0]
        return self._index.search(vector, pool)

    def _keyword_ranked(self, query: str, pool: int) -> list[tuple[str, float]]:
        if self._custom_keyword is not None:
            return self._custom_keyword(query, pool)
        assert self._keyword is not None  # noqa: S101 - guarded by search()
        return self._keyword.search(query, pool)