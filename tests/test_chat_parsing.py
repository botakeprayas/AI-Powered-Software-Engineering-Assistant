import pytest
from chat_helpers import make_evidence

from core.chat.parsing import build_answer, extract_ids, parse_model_output, strip_citations
from services.llm_service import LLMResponse


def respond(text, truncated=False):
    return LLMResponse(text=text, model="m", provider="test", truncated=truncated)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a [E1]", ["E1"]),
        ("a [E1][E3]", ["E1", "E3"]),
        ("a [E1, E2]", ["E1", "E2"]),
        ("a [e2]", ["E2"]),
        ("a [E1] b [E1]", ["E1"]),
        ("no citation", []),
    ],
)
def test_extract_ids(text, expected):
    assert extract_ids(text) == expected


def test_strip_citations_cleans_spacing():
    assert strip_citations("It charges the gateway [E1][E2].") == "It charges the gateway."


WELL_FORMED = """STATUS: ANSWERED
FACTS:
- checkout charges the gateway [E1]
* total multiplies price by quantity [E2]
1. numbered item [E1, E2]
INFERENCES:
- The cart probably supports payment [E1][E2]
UNKNOWNS:
- none
"""


def test_parse_well_formed_output():
    parsed = parse_model_output(WELL_FORMED)
    assert parsed.recognized
    assert parsed.status == "answered"
    assert len(parsed.facts) == 3
    assert len(parsed.inferences) == 1
    assert parsed.unknowns == []


def test_parse_markdown_headings_and_wrapped_lines():
    text = (
        "**STATUS:** PARTIAL\n**FACTS:**\n- first part\n  continues here [E1]\n"
        "## UNKNOWNS\n- missing"
    )
    parsed = parse_model_output(text)
    assert parsed.status == "partial"
    assert parsed.facts == ["first part continues here [E1]"]
    assert parsed.unknowns == ["missing"]


def test_free_text_is_not_recognized():
    assert not parse_model_output("Sure! The checkout function charges a card.").recognized


def test_build_answer_attaches_indexed_locations():
    answer = build_answer("q", respond(WELL_FORMED), make_evidence(2))
    assert answer.status == "answered"
    assert [c.evidence_ids for c in answer.facts] == [("E1",), ("E2",), ("E1", "E2")]
    assert [e.location for e in answer.citations()] == ["shop/cart.py:22-27", "shop/cart.py:18-20"]


def test_unknown_id_is_removed_and_reported():
    text = "STATUS: ANSWERED\nFACTS:\n- real claim [E1][E9]\nINFERENCES:\n- none\nUNKNOWNS:\n- none"
    answer = build_answer("q", respond(text), make_evidence(2))
    assert answer.facts[0].evidence_ids == ("E1",)
    assert any("E9" in w for w in answer.warnings)


def test_uncited_statement_is_dropped_and_status_downgraded():
    text = (
        "STATUS: ANSWERED\nFACTS:\n- cited [E1]\n- not cited\n"
        "INFERENCES:\n- none\nUNKNOWNS:\n- none"
    )
    answer = build_answer("q", respond(text), make_evidence(2))
    assert [c.text for c in answer.facts] == ["cited"]
    assert answer.status == "partial"
    assert any("without a valid citation" in w for w in answer.warnings)


def test_no_valid_citation_at_all_means_insufficient():
    text = "STATUS: ANSWERED\nFACTS:\n- not cited\nINFERENCES:\n- none\nUNKNOWNS:\n- none"
    answer = build_answer("q", respond(text), make_evidence(2))
    assert answer.status == "insufficient"
    assert answer.facts == [] and answer.inferences == []


def test_model_abstention_is_respected_and_facts_ignored():
    text = (
        "STATUS: INSUFFICIENT_EVIDENCE\nFACTS:\n- sneaky [E1]\n"
        "INFERENCES:\n- none\nUNKNOWNS:\n- no payment code"
    )
    answer = build_answer("q", respond(text), make_evidence(2))
    assert answer.status == "insufficient"
    assert answer.facts == []
    assert answer.unknowns == ["no payment code"]


def test_unstructured_text_is_rejected_but_kept_for_debugging():
    answer = build_answer("q", respond("It works by magic."), make_evidence(2))
    assert answer.status == "insufficient"
    assert answer.raw_text == "It works by magic."
    assert any("required answer format" in w for w in answer.warnings)


def test_truncated_response_adds_warning():
    answer = build_answer("q", respond(WELL_FORMED, truncated=True), make_evidence(2))
    assert any("length limit" in w for w in answer.warnings)