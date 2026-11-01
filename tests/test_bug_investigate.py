import pytest
from bug_helpers import (
    BUG_SAMPLE,
    FABRICATED_REPLY,
    FIX_REPLY,
    HYPOTHESIS_REPLY,
    TEST_REPLY,
    UNSUPPORTED_HYPOTHESIS_REPLY,
    SequenceLLM,
    sample_context,
    sample_traceback,
    validation_config,
)

from core.bugs.investigate import investigate
from core.bugs.proposal import DRAFT_NONE, DRAFT_READY, FIX_PROPOSED
from core.validation.models import LEVEL_NOT_APPLICABLE, LEVEL_SUFFICIENT, SUPPORTED, UNSUPPORTED
from services.llm_service import LLMConnectionError, MockLLM


@pytest.fixture(scope="module")
def context():
    return sample_context()


def run(context, llm, text=None, **kwargs):
    return investigate(
        text or sample_traceback(), context, llm, validation_config=validation_config(), **kwargs
    )


def snapshot_files():
    return {str(p): p.read_bytes() for p in BUG_SAMPLE.rglob("*") if p.is_file()}


def test_full_flow_gives_a_validated_hypothesis_a_proposed_fix_and_a_test_draft(context):
    before = snapshot_files()
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    inv = run(context, llm)
    assert inv.errors == []
    assert inv.hypothesis.status == "answered"
    assert [c.final for c in inv.report.claims] == [SUPPORTED, SUPPORTED]
    assert inv.report.policy.level == LEVEL_SUFFICIENT
    assert inv.fix.status == FIX_PROPOSED and inv.fix.label == "PROPOSED, NOT VERIFIED"
    assert "+    if not items:" in inv.fix.diff
    assert inv.test_draft.status == DRAFT_READY
    assert inv.test_draft.path == "tests/test_regression_average_price.py"
    assert snapshot_files() == before  # nothing was written to the project


def test_hypothesis_links_to_indexed_source_evidence(context):
    inv = run(context, SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY]))
    cited = {item.location for item in inv.hypothesis.citations()}
    assert cited == {"cart/pricing.py:4-7", "cart/checkout.py:5-8"}
    assert all(check.ok for check in inv.report.citations)


def test_prompts_carry_the_evidence_and_keep_fix_code_plain(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    run(context, llm)
    hypothesis, fix, test = llm.prompts
    assert "TRACEBACK (untrusted text)" in hypothesis
    assert "[E1] cart/pricing.py:4-7 | function | average_price | failing frame:" in hypothesis
    assert "-> E1 (average_price)" in hypothesis
    assert "-> not mapped (the file is not in the indexed repository)" in hypothesis
    assert "via call graph, caller of E1" in hypothesis
    assert "CALL GRAPH CONTEXT" in llm.systems[0]
    assert "Hypothesis: `average_price` divides by `len(items)`" in fix
    assert "return total / len(items)" in fix and "<--" not in fix
    assert "test_average_of_two_items" not in fix
    assert "- cart/pricing.py" in fix and "tests/test_pricing.py" not in fix.split("EXCERPTS")[0]
    assert "+    if not items:" in test and "test_average_of_two_items" in test


def test_fabricated_citations_leave_nothing_to_validate_and_no_fix(context):
    llm = SequenceLLM([FABRICATED_REPLY])
    inv = run(context, llm)
    assert inv.hypothesis.status == "insufficient"
    assert inv.report.policy.level == LEVEL_NOT_APPLICABLE
    assert inv.fix is None and inv.test_draft is None
    assert len(llm.prompts) == 1
    assert any("No fix was proposed" in step for step in inv.steps)


def test_an_unsupported_hypothesis_is_flagged_and_does_not_generate_a_fix(context):
    llm = SequenceLLM([UNSUPPORTED_HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    inv = run(context, llm)
    assert [c.final for c in inv.report.claims] == [SUPPORTED, UNSUPPORTED]
    assert any("refund_all_orders" in reason for reason in inv.report.claims[1].reasons)
    assert inv.fix is None and len(llm.prompts) == 1
    assert any("no root-cause hypothesis passed validation" in step for step in inv.steps)


def test_llm_errors_become_messages_not_crashes(context):
    error = LLMConnectionError("Cannot reach Ollama")
    first = run(context, SequenceLLM([HYPOTHESIS_REPLY], errors={1: error}))
    assert first.hypothesis is None and "Cannot reach Ollama" in first.errors[0]
    second = run(context, SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY], errors={2: error}))
    assert second.hypothesis is not None and second.fix is None
    assert any("Fix step" in text for text in second.errors)
    assert second.test_draft.status == DRAFT_READY  # the test step still ran, without a diff


def test_the_mock_provider_skips_fix_and_test_generation(context):
    inv = run(context, MockLLM())
    assert inv.is_mock and inv.hypothesis is not None
    assert inv.fix is None and inv.test_draft is None
    assert any("MOCK" in step for step in inv.steps)


def test_inputs_with_nothing_to_investigate(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY])
    none = run(context, llm, text="just some words")
    assert none.errors and none.hypothesis is None
    elsewhere = 'File "elsewhere.py", line 3, in f\n    x\nValueError: bad\n'
    unmapped = run(context, llm, text=elsewhere)
    assert any("No traceback frame maps" in text for text in unmapped.errors)
    assert llm.prompts == []

RISKY_TEST_REPLY = (
    "FILE: tests/test_regression_x.py\n```python\n"
    "import socket\n\n\ndef test_x():\n    socket.socket()\n```\n"
)


def test_a_declined_test_is_retried_once_with_the_reason(context):
    declined = "NO_TEST: no test covers the empty case"
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, declined, TEST_REPLY])
    inv = run(context, llm)
    assert inv.test_draft.status == DRAFT_READY
    assert len(llm.prompts) == 4
    assert "YOUR PREVIOUS ANSWER WAS NOT USABLE" in llm.prompts[3]
    assert "no test covers the empty case" in llm.prompts[3]
    assert any("asked once more" in step for step in inv.steps)


def test_a_second_refusal_is_kept_and_there_is_no_third_attempt(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, "NO_TEST: first", "NO_TEST: second"])
    inv = run(context, llm)
    assert inv.test_draft.status == DRAFT_NONE
    assert inv.test_draft.reason == "second"
    assert len(llm.prompts) == 4


def test_a_rejected_draft_is_retried_with_the_problems(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, RISKY_TEST_REPLY, TEST_REPLY])
    inv = run(context, llm)
    assert inv.test_draft.status == DRAFT_READY
    assert "imports socket" in llm.prompts[3]


def test_an_error_during_the_retry_keeps_the_first_draft(context):
    error = LLMConnectionError("Cannot reach Ollama")
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, "NO_TEST: first"], errors={4: error})
    inv = run(context, llm)
    assert inv.test_draft.status == DRAFT_NONE and inv.test_draft.reason == "first"
    assert any("Test step" in text for text in inv.errors)


def test_the_prompts_forbid_the_mistakes_seen_with_a_real_model(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    run(context, llm)
    assert "callers" in llm.systems[1]
    assert "NOT a reason to decline" in llm.systems[2]
    assert "OUTERMOST repository function" in llm.systems[2]

def test_the_test_prompt_shows_the_failing_call_and_the_retry_cannot_decline(context):
    llm = SequenceLLM([HYPOTHESIS_REPLY, FIX_REPLY, "NO_TEST: not needed", TEST_REPLY])
    inv = run(context, llm)
    assert inv.test_draft.status == DRAFT_READY
    assert "THE CALL THAT FAILED" in llm.prompts[2]
    assert "print(summary([]))" in llm.prompts[2]
    assert "NO_TEST" in llm.systems[2]
    assert "NO_TEST" not in llm.systems[3]
    assert "MUST write the test" in llm.systems[3]