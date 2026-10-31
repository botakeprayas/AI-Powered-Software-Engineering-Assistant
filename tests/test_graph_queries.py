import pytest
from graph_helpers import key_of, sample_context

from core.graph.models import NODE_FILE


@pytest.fixture(scope="module")
def context():
    return sample_context()


@pytest.fixture(scope="module")
def graph(context):
    return context.graph


def where(relations):
    return sorted((r.source_label, r.target_label, r.location) for r in relations)


def test_callees_of_checkout(context, graph):
    key = key_of(context, "OrderService.checkout")
    assert where(graph.callees_of(key)) == [
        ("OrderService.checkout", "OrderService.log_total", "orders/service.py:20"),
        ("OrderService.checkout", "OrderService.subtotal", "orders/service.py:18"),
        ("OrderService.checkout", "apply_tax", "orders/service.py:18"),
    ]


def test_callers_of_apply_tax(context, graph):
    key = key_of(context, "apply_tax")
    assert where(graph.callers_of(key)) == [
        ("OrderService.checkout", "apply_tax", "orders/service.py:18")
    ]


def test_callers_of_a_class_are_its_instantiations(context, graph):
    key = key_of(context, "OrderService")
    assert where(graph.callers_of(key)) == [
        ("place_order", "OrderService", "orders/service.py:30"),
        ("run", "OrderService", "main.py:9"),
    ]


def test_module_level_code_can_be_a_caller(context, graph):
    (rel,) = graph.callers_of(key_of(context, "run"))
    assert graph.node(rel.source)["kind"] == NODE_FILE
    assert rel.location == "main.py:16"


def test_uncertain_callees_are_listed_separately(context, graph):
    (rel,) = graph.uncertain_callees_of(key_of(context, "OrderService.checkout"))
    assert rel.target_label == "self.gateway.charge"
    assert rel.location == "orders/service.py:19"


def test_possible_callers_are_not_confirmed_callers(context, graph):
    for qualname in ("StripeGateway.charge", "MockGateway.charge"):
        key = key_of(context, qualname)
        assert graph.callers_of(key) == []
        assert [r.source_label for r in graph.possible_callers_of(key)] == [
            "OrderService.checkout"
        ]


def test_unknown_nodes_have_no_relations(graph):
    assert graph.callers_of("nope.py::nothing") == []
    assert graph.callees_of("nope.py::nothing") == []


def test_find_symbols_by_short_or_qualified_name(graph):
    assert graph.find_symbols("checkout") == ["orders/service.py::OrderService.checkout"]
    assert len(graph.find_symbols("charge")) == 2