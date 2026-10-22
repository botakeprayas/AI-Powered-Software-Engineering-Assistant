"""Reciprocal Rank Fusion (RRF): a simple, transparent way to merge ranked lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

RRF_K = 60  # standard default; larger = flatter, smaller = top ranks dominate


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Each list gives an item weight / (k + rank) points; points are summed across lists."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking, start=1):
            scores[key] += weight / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))