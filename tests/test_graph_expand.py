import dataclasses

import pytest
from graph_helpers import key_of, sample_context, seed_for

from core.chat.models import QAConfig
from core.chat.qa import TRUNCATION_MARKER
from core.graph.expand import (
    ExpansionConfig,
    GraphEvidence,
    expand_evidence,
    load_expansion_config,
)

QA = QAConfig()
CHECKOUT = "OrderService.checkout"
ENV_NAMES = (
    "GRAPH_EXPAND_DEPTH", "GRAPH_EXPAND_MAX_PER_DIRECTION", "GRAPH_EXPAND_MAX_ADDED",
    "GRAPH_EXPAND_MAX_SEEDS",
)


@pytest.fixture(scope="module")
def context():
    return sample_context()


def names(result):
    return {e.symbol.qualname for e in result.evidence}


def test_connected_code_from_other_files_is_added_with_reasons(context):
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], QA, ExpansionConfig())
    assert names(result) == {"OrderService.subtotal", "apply_tax", "run", "place_order"}
    assert sorted(e.id for e in result.evidence) == ["E2", "E3", "E4", "E5"]
    for item in result.evidence:
        assert isinstance(item, GraphEvidence)
        assert item.seed_id == "E1" and "of E1" in item.via
        assert item.relations and item.relations[0].file in item.via
    apply_tax = next(e for e in result.evidence if e.symbol.qualname == "apply_tax")
    assert apply_tax.direction == "callee"
    assert "orders/service.py:18" in apply_tax.via
    assert apply_tax.location == "utils/helpers.py:9-11"


def test_uncertain_relationships_become_notes_never_evidence(context):
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], QA, ExpansionConfig())
    assert not [n for n in names(result) if n.endswith("charge")]
    note = next(n for n in result.notes if "self.gateway.charge" in n)
    assert "UNCERTAIN" in note and "NOT confirmed" in note
    assert "StripeGateway.charge" in note


def test_the_added_excerpt_limit_is_respected_and_reported(context):
    config = ExpansionConfig(max_added=1)
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], QA, config)
    assert len(result.evidence) == 1
    assert any("limit" in text for text in result.skipped)


def test_depth_two_reaches_further_and_records_every_hop(context):
    big = {"max_per_direction": 10, "max_added": 30, "max_total_chars": 10**6}
    seed = seed_for(context, "run")
    one = expand_evidence(context, [seed], QA, ExpansionConfig(depth=1, **big))
    two = expand_evidence(context, [seed], QA, ExpansionConfig(depth=2, **big))
    assert "OrderService.subtotal" not in names(one)
    assert names(one) < names(two)
    deep = next(e for e in two.evidence if e.symbol.qualname == "OrderService.subtotal")
    assert deep.depth == 2
    assert "run calls OrderService.checkout" in deep.via
    assert "OrderService.checkout calls OrderService.subtotal" in deep.via


def test_module_level_callers_become_a_note(context):
    result = expand_evidence(context, [seed_for(context, "run")], QA, ExpansionConfig())
    assert any("module-level" in n and "main.py:16" in n for n in result.notes)


def test_retrieved_code_is_not_added_twice(context):
    seeds = [seed_for(context, CHECKOUT, "E1"), seed_for(context, "OrderService.subtotal", "E2")]
    result = expand_evidence(context, seeds, QA, ExpansionConfig())
    assert "OrderService.subtotal" not in names(result)
    assert min(e.id for e in result.evidence) == "E3"


def test_a_seed_missing_from_the_graph_is_skipped_with_a_reason(context):
    seed = seed_for(context, CHECKOUT)
    ghost = dataclasses.replace(seed.symbol, key="ghost.py::ghost")
    seed = dataclasses.replace(seed, symbol=ghost)
    result = expand_evidence(context, [seed], QA, ExpansionConfig())
    assert result.evidence == []
    assert any("not in the dependency graph" in text for text in result.skipped)


def test_the_prompt_size_budget_is_enforced(context):
    config = ExpansionConfig(max_total_chars=10)
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], QA, config)
    assert result.evidence == []
    assert any("prompt size limit" in text for text in result.skipped)


def test_added_code_is_truncated_like_phase_3_code(context):
    small = dataclasses.replace(QA, max_code_chars=20)
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], small, ExpansionConfig())
    assert result.evidence
    assert all(e.code_truncated and e.code.endswith(TRUNCATION_MARKER) for e in result.evidence)


def test_every_added_excerpt_points_at_its_indexed_location(context):
    result = expand_evidence(context, [seed_for(context, CHECKOUT)], QA, ExpansionConfig())
    for item in result.evidence:
        sym = context.symbols[item.symbol.key]
        assert item.location == f"{sym.file}:{sym.start_line}-{sym.end_line}"
        assert key_of(context, sym.qualname) == sym.key


@pytest.mark.parametrize("depth", [0, 4])
def test_invalid_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="GRAPH_EXPAND_DEPTH"):
        ExpansionConfig(depth=depth)


def test_config_is_read_from_the_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert load_expansion_config(env_file=None) == ExpansionConfig()
    monkeypatch.setenv("GRAPH_EXPAND_DEPTH", "2")
    assert load_expansion_config(env_file=None).depth == 2
    monkeypatch.setenv("GRAPH_EXPAND_DEPTH", "9")
    with pytest.raises(ValueError, match="GRAPH_EXPAND_DEPTH"):
        load_expansion_config(env_file=None)
    monkeypatch.setenv("GRAPH_EXPAND_DEPTH", "abc")
    with pytest.raises(ValueError, match="whole number"):
        load_expansion_config(env_file=None)