import pytest
from graph_helpers import GRAPH_SAMPLE, sample_context

from core.graph.models import (
    CALLS,
    CONFIRMED,
    EDGE_KINDS,
    IMPORTS,
    NODE_UNRESOLVED,
    UNCERTAIN,
)


@pytest.fixture(scope="module")
def graph():
    return sample_context().graph


def test_counts_match_the_sample(graph):
    stats = graph.stats
    assert stats.nodes == {"file": 8, "function": 12, "class": 3, "module": 1, "unresolved": 4}
    assert stats.edges == {
        "imports/confirmed": 12,
        "defines/confirmed": 15,
        "calls/confirmed": 12,
        "calls/uncertain": 4,
    }
    assert stats.ignored == {"builtin": 6}


def test_symbol_node_ids_are_the_phase_1_keys(graph):
    assert graph.has_node("orders/service.py::OrderService.checkout")
    assert graph.has_node("utils/helpers.py::apply_tax")


def test_every_relationship_traces_to_a_real_source_line(graph):
    relations = graph.relations()
    assert relations
    for rel in relations:
        assert rel.kind in EDGE_KINDS
        assert rel.certainty in (CONFIRMED, UNCERTAIN)
        assert rel.reason
        lines = (GRAPH_SAMPLE / rel.file).read_text(encoding="utf-8").split("\n")
        assert 1 <= rel.line <= len(lines), rel
        text = lines[rel.line - 1]
        if rel.kind == CALLS:
            assert "(" in text or text.lstrip().startswith("@"), (rel.location, text)
        elif rel.kind == IMPORTS:
            assert "import" in text, (rel.location, text)
        else:
            assert text.lstrip().startswith(("def ", "async def ", "class ", "@")), text


def test_dynamic_calls_are_flagged_uncertain_not_guessed(graph):
    uncertain = {
        (r.source_label, r.target_label, r.location)
        for r in graph.relations(CALLS, UNCERTAIN)
    }
    assert ("run_plugin", "importlib.import_module", "plugins/loader.py:7") in uncertain
    assert ("run_plugin", "handler", "plugins/loader.py:9") in uncertain
    assert ("apply_callback", "callback", "utils/helpers.py:16") in uncertain
    confirmed = graph.relations(CALLS, CONFIRMED)
    assert not [r for r in confirmed if r.source_label == "run_plugin"]


def test_uncertain_calls_end_in_unresolved_nodes_never_in_real_code(graph):
    for rel in graph.relations(CALLS, UNCERTAIN):
        assert graph.node(rel.target)["kind"] == NODE_UNRESOLVED
    for rel in graph.relations(CALLS, CONFIRMED):
        assert graph.node(rel.target)["kind"] != NODE_UNRESOLVED


def test_candidates_are_never_turned_into_confirmed_edges(graph):
    charge = [rel for rel in graph.relations(CALLS, UNCERTAIN) if "charge" in rel.target_label]
    assert len(charge) == 1
    assert {graph.label(c) for c in charge[0].candidates} == {
        "StripeGateway.charge",
        "MockGateway.charge",
    }
    for key in charge[0].candidates:
        assert graph.callers_of(key) == []