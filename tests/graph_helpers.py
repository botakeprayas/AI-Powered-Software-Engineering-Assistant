"""Helpers for the Phase 4 tests."""

from __future__ import annotations

from pathlib import Path

from core.chat.models import Evidence
from core.graph.builder import build_graph
from core.graph.expand import GraphContext, load_graph_context
from core.graph.graph import DependencyGraph
from core.repo.pipeline import analyze_repository

GRAPH_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "graph_sample"


def sample_context() -> GraphContext:
    return load_graph_context(GRAPH_SAMPLE)


def key_of(context: GraphContext, qualname: str) -> str:
    matches = [key for key, sym in context.symbols.items() if sym.qualname == qualname]
    assert len(matches) == 1, f"{qualname}: {matches}"
    return matches[0]


def seed_for(context: GraphContext, qualname: str, eid: str = "E1") -> Evidence:
    symbol = context.symbols[key_of(context, qualname)]
    return Evidence(
        id=eid, symbol=symbol, code=symbol.code, code_truncated=False,
        semantic_score=0.55, keyword_rank=1, keyword_score=12.0,
    )


def graph_of(root: Path, files: dict[str, str]) -> DependencyGraph:
    """Write a tiny repository into `root` and build its dependency graph."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    analysis = analyze_repository(root)
    try:
        return build_graph(analysis)
    finally:
        analysis.close()


def calls(graph: DependencyGraph) -> set[tuple[str, str, str]]:
    """(caller label, target label, certainty) for every call relationship."""
    return {(r.source_label, r.target_label, r.certainty) for r in graph.relations("calls")}