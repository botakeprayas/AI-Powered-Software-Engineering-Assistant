import pytest
from graph_helpers import key_of, sample_context

from core.graph.models import CALLS, CONFIRMED, UNCERTAIN
from core.graph.viz import GraphView, draw_view, figure_png, select_view

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def context():
    return sample_context()


@pytest.fixture(scope="module")
def graph(context):
    return context.graph


def test_whole_graph_view_contains_every_call_relationship(graph):
    view = select_view(graph)
    assert len(view.relations) == 16
    assert not view.truncated
    assert all(rel.kind == CALLS for rel in view.relations)


def test_focus_view_shows_the_neighbours_and_nothing_unrelated(context, graph):
    focus = key_of(context, "OrderService.checkout")
    view = select_view(graph, focus=focus, depth=1)
    labels = {graph.label(n) for n in view.nodes}
    assert {"OrderService.subtotal", "apply_tax", "run", "place_order"} <= labels
    assert "self.gateway.charge" in labels  # the unresolved placeholder, drawn dashed
    assert "run_plugin" not in labels


def test_certainty_filter(graph):
    view = select_view(graph, certainties=(CONFIRMED,))
    assert view.relations and all(rel.certainty == CONFIRMED for rel in view.relations)
    only_uncertain = select_view(graph, certainties=(UNCERTAIN,))
    assert len(only_uncertain.relations) == 4


def test_node_cap_truncates_the_view(graph):
    view = select_view(graph, max_nodes=5)
    assert len(view.nodes) == 5
    assert view.truncated
    keep = set(view.nodes)
    assert all(rel.source in keep and rel.target in keep for rel in view.relations)


def test_unknown_focus_is_rejected(graph):
    with pytest.raises(KeyError):
        select_view(graph, focus="nope.py::nothing")


def test_the_figure_renders_to_a_png(graph):
    png = figure_png(draw_view(graph, select_view(graph), "sample"))
    assert png.startswith(PNG) and len(png) > 1000


def test_an_empty_view_still_renders(graph):
    assert figure_png(draw_view(graph, GraphView())).startswith(PNG)