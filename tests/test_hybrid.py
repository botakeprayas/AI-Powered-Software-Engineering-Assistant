import pytest

from core.semantic.fusion import reciprocal_rank_fusion


def test_rrf_item_in_both_lists_wins():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0][0] == "b"


def test_rrf_respects_weights():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])
    assert fused[0][0] == "b"


@pytest.mark.parametrize(
    ("query", "expected"),
    [("compute_invoice_total", "compute_invoice_total"), ("ShoppingCart.add_item", "add_item")],
)
def test_exact_identifier_keyword_search_needs_no_model(
    engine, fake_embedder, query, expected
):
    engine.sync(semantic=False)
    results = engine.search(query, mode="keyword", top_k=3)
    assert results[0].symbol.name == expected
    assert fake_embedder.texts_embedded == 0  # proves the model was never used


def test_hybrid_result_exposes_both_ranks(engine):
    engine.sync()
    results = engine.search("compute_invoice_total", mode="hybrid", top_k=3)
    top = next(r for r in results if r.symbol.name == "compute_invoice_total")
    assert top.keyword_rank == 1
    assert top.semantic_rank is not None


def test_search_before_sync_raises(engine):
    with pytest.raises(RuntimeError):
        engine.search("anything")


def test_empty_query_is_rejected(engine):
    engine.sync(semantic=False)
    with pytest.raises(ValueError):
        engine.search("   ", mode="keyword")