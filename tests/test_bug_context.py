from pathlib import Path

import pytest
from bug_helpers import sample_context, sample_traceback

from core.bugs.context import (
    BugContextConfig,
    build_bug_context,
    find_related_tests,
    window_code,
)
from core.bugs.mapping import FrameMapper
from core.bugs.traceback_parser import parse_traceback
from core.chat.models import QAConfig
from core.graph.expand import GraphContext, GraphEvidence
from core.graph.models import Relation
from core.semantic.models import Symbol


@pytest.fixture(scope="module")
def context():
    return sample_context()


@pytest.fixture(scope="module")
def parsed_and_mapped(context):
    parsed = parse_traceback(sample_traceback())
    mapper = FrameMapper(list(context.symbols.values()), root_hint=str(context.root))
    return parsed, mapper.map(parsed.frames)


@pytest.fixture(scope="module")
def bug(context, parsed_and_mapped):
    parsed, mapped = parsed_and_mapped
    return build_bug_context(parsed, mapped, context, QAConfig())


def test_frames_map_to_the_failing_function_its_caller_and_one_refusal(parsed_and_mapped):
    _, mapped = parsed_and_mapped
    got = [(m.status, m.symbol.qualname if m.symbol else None) for m in mapped]
    assert got == [("unindexed", None), ("mapped", "summary"), ("mapped", "average_price")]


def test_evidence_has_consecutive_ids_roles_and_reasons(bug):
    evidence = bug.evidence
    assert [e.id for e in evidence] == ["E1", "E2", "E3"]
    names = [e.symbol.qualname for e in evidence]
    assert names == ["average_price", "summary", "test_average_of_two_items"]
    assert [e.role for e in evidence[:2]] == ["failing frame", "calling frame"]
    assert isinstance(evidence[2], GraphEvidence)
    assert "caller of E1" in evidence[2].via
    assert "tests/test_pricing.py:5" in evidence[2].via


def test_the_failing_line_is_marked_in_the_excerpt_but_not_in_the_plain_code(bug):
    first = bug.evidence[0]
    marked = "return total / len(items)  # <-- failing line: ZeroDivisionError: division by zero"
    assert marked in first.code
    assert "<--" not in first.plain_code
    assert first.plain_code == first.symbol.code
    assert bug.evidence[1].code.count("<-- traceback frame") == 1


def test_refused_frames_are_listed_with_reasons(bug):
    assert [(m.status, m.frame.file) for m in bug.refused] == [("unindexed", "demo.py")]
    assert "not in the indexed repository" in bug.refused[0].reason


def test_test_ids_and_editable_files(bug):
    assert bug.related_test_ids() == ["tests/test_pricing.py::test_average_of_two_items"]
    assert bug.editable_files() == ["cart/checkout.py", "cart/pricing.py"]
    assert bug.primary.symbol.qualname == "average_price"


def test_no_mapped_frames_gives_no_evidence_and_a_note(context):
    parsed = parse_traceback(sample_traceback())
    result = build_bug_context(parsed, [], context, QAConfig())
    assert result.evidence == [] and result.refused == []
    assert result.notes


def make_symbol(file, qualname, start, code, kind="function"):
    return Symbol(
        key=f"{file}::{qualname}", file=file, kind=kind, name=qualname.split(".")[-1],
        qualname=qualname, start_line=start, end_line=start + code.count("\n"),
        docstring="", code=code,
    )


def test_window_code_cuts_around_the_failing_line_and_marks_it():
    body = "\n".join(f"    step_{i} = {i}" for i in range(60))
    symbol = make_symbol("m.py", "long_one", 10, "def long_one():\n" + body)
    marked, plain, cut = window_code(symbol, 40, 300, "mark")
    assert cut
    assert "step_29 = 29  # <-- mark" in marked
    assert "<--" not in plain
    assert "line(s) omitted" in marked and len(plain) < 450


def test_short_code_is_not_cut():
    symbol = make_symbol("m.py", "short", 1, "def short():\n    return 1")
    marked, plain, cut = window_code(symbol, 2, 1500, "mark")
    assert not cut and plain == symbol.code
    assert marked.endswith("return 1  # <-- mark")


class FakeGraph:
    def __init__(self, callers=None, error=False):
        self._callers = callers or {}
        self._error = error

    def callers_of(self, key):
        if self._error:
            raise RuntimeError("graph exploded")
        return self._callers.get(key, [])


def related_setup():
    target = make_symbol("shop/cart.py", "total", 3, "def total(self):\n    return 1")
    near = make_symbol("tests/test_cart.py", "test_total", 1, "def test_total():\n    cart.total()")
    other = make_symbol("tests/test_other.py", "test_misc", 1, "def test_misc():\n    assert 1")
    helper = make_symbol("tests/helpers.py", "make_cart", 1, "def make_cart():\n    total()")
    symbols = {s.key: s for s in (target, near, other, helper)}
    return target, near, symbols


def test_related_tests_are_scored_and_explained():
    target, near, symbols = related_setup()
    context = GraphContext(FakeGraph(), symbols, Path("."))
    found = find_related_tests(context, [target], set(), 2)
    assert [(s.qualname, len(r)) for s, r in found] == [("test_total", 2)]
    reasons = " ".join(found[0][1])
    assert "named after shop/cart.py" in reasons and "mentions total" in reasons


def test_related_tests_use_confirmed_graph_callers_and_survive_a_broken_graph():
    target, near, symbols = related_setup()
    rel = Relation(
        "calls", "confirmed", near.key, target.key, "test_total", "total",
        "tests/test_cart.py", 2, "defined in this file",
    )
    graph = FakeGraph({target.key: [rel]})
    found = find_related_tests(GraphContext(graph, symbols, Path(".")), [target], set(), 2)
    assert "calls total at tests/test_cart.py:2 (confirmed)" in found[0][1]
    broken = GraphContext(FakeGraph(error=True), symbols, Path("."))
    assert [s.qualname for s, _ in find_related_tests(broken, [target], set(), 2)] == ["test_total"]
    assert find_related_tests(broken, [target], {near.key}, 2) == []


def test_the_related_test_limit_is_respected(context):
    target, near, symbols = related_setup()
    code = "def test_total_two():\n    total()"
    extra = make_symbol("tests/test_cart.py", "test_total_two", 5, code)
    symbols[extra.key] = extra
    found = find_related_tests(GraphContext(FakeGraph(), symbols, Path(".")), [target], set(), 1)
    assert len(found) == 1
    assert BugContextConfig().max_tests == 2