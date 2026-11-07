"""Helpers for the Phase 8 indexing tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from memory_helpers import copy_sample

from core.graph.builder import build_graph
from core.memory.indexer import IncrementalIndexer, load_graph, load_symbols
from core.memory.store import MemoryStore
from core.repo.pipeline import analyze_repository

RATES = "pricing/rates.py"
ALL_FILES = [
    "app.py", "pricing/__init__.py", "pricing/invoice.py", "pricing/rates.py",
    "shop/__init__.py", "shop/checkout.py", "shop/display.py", "shop/receipt.py",
    "tests/test_checkout.py", "tests/test_invoice.py", "tests/test_rates.py",
    "tools/formatting.py",
]


def parseable(root, paths):
    """The files that have content. An empty file is recorded but never parsed for the
    graph (exactly as in the Phase 4 builder), so this is what the parse spy should see."""
    return sorted(p for p in paths if (Path(root) / p).read_bytes().strip())


def scratch(root):
    """The Phase 1 + Phase 4 result built from scratch (the reference answer)."""
    analysis = analyze_repository(root)
    try:
        return build_graph(analysis), list(analysis.symbols)
    finally:
        analysis.close()


def canonical(graph):
    """A comparable form of a graph: nodes, edges (with reasons) and statistics."""
    nodes = {node_id: dict(data) for node_id, data in graph.nx_graph.nodes(data=True)}
    edges = sorted(
        (s, t, d["kind"], d["certainty"], d["file"], d["line"], d["reason"],
         tuple(d["candidates"]))
        for s, t, d in graph.nx_graph.edges(data=True)
    )
    stats = graph.stats
    return nodes, edges, (
        dict(stats.nodes), dict(stats.edges), dict(stats.ignored),
        stats.files_analyzed, stats.files_skipped,
    )


def assert_matches_scratch(store, root):
    """The incrementally maintained index must equal a build from scratch."""
    graph, symbols = scratch(root)
    assert load_symbols(store) == symbols
    assert canonical(load_graph(store)) == canonical(graph)


def indexed(tmp_path, embedder=None):
    """A copy of impact_sample, indexed once. Returns the pieces tests need."""
    root = copy_sample(tmp_path / "repo")
    store = MemoryStore()
    indexer = IncrementalIndexer(root, store)
    db = tmp_path / "semantic.db"
    first = indexer.update(embedder=embedder, semantic_db=db if embedder else None)
    return SimpleNamespace(root=root, store=store, indexer=indexer, first=first, db=db)