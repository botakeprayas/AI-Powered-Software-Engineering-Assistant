import dataclasses

import pytest
from repo_helpers import SAMPLE_REPO
from validation_helpers import (
    ScriptedJudge,
    copy_sample,
    evidence_for,
    load_symbols,
    make_answer,
    snapshot_for,
)

from core.chat.models import STATUS_INSUFFICIENT, QAConfig
from core.semantic.models import SearchResult
from core.validation.config import ValidationConfig
from core.validation.models import (
    CITE_VALID,
    JUDGE_SKIPPED,
    JUDGE_UNPARSED,
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    LEVEL_SUFFICIENT,
    SUPPORTED,
    UNSUPPORTED,
)
from core.validation.report import candidate_rows, evidence_rows, validate_answer
from services.llm_service import LLMConnectionError

CONFIG = ValidationConfig()
GOOD = "`checkout` awaits `payment_gateway.charge`"


@pytest.fixture(scope="module")
def symbols():
    return load_symbols()


def build(symbols, facts, inferences=(), root=SAMPLE_REPO):
    evidence = [evidence_for(symbols["ShoppingCart.checkout"])]
    answer = make_answer(facts=facts, inferences=inferences, evidence=evidence)
    return answer, snapshot_for(root, symbols)


def run(symbols, facts, judge=None, config=CONFIG):
    answer, snap = build(symbols, facts)
    return validate_answer(answer, snap, config, judge)


def test_supported_claim_with_a_judge(symbols):
    judge = ScriptedJudge("F1: SUPPORTED | shown on line 25")
    report = run(symbols, [(GOOD, ["E1"])], judge)
    claim = report.claims[0]
    assert (claim.final, claim.basis) == (SUPPORTED, "judge + rules")
    assert report.policy.level == LEVEL_SUFFICIENT
    assert report.counts()["claims_supported"] == 1


def test_judge_rejection_flags_the_claim(symbols):
    facts = [("`checkout` empties the cart", ["E1"])]
    report = run(symbols, facts, ScriptedJudge("F1: NOT_SUPPORTED | not shown"))
    assert report.claims[0].final == UNSUPPORTED
    assert report.policy.level == LEVEL_LIMITED


def test_name_missing_from_cited_code_is_unsupported(symbols):
    text = "`checkout` calls `payment_gateway.refund`"
    report = run(symbols, [(text, ["E1"])])
    claim = report.claims[0]
    assert claim.final == UNSUPPORTED
    assert any("payment_gateway.refund" in reason for reason in claim.reasons)
    assert report.policy.level == LEVEL_LIMITED


def test_invented_location_flags_the_claim(symbols):
    text = "`checkout` charges the gateway at shop/cart.py:90-120"
    report = run(symbols, [(text, ["E1"])], ScriptedJudge("F1: SUPPORTED | ok"))
    claim = report.claims[0]
    assert claim.final == UNSUPPORTED
    assert any("invented location" in reason for reason in claim.reasons)
    assert any("shop/cart.py:90-120" in text for text in report.policy.limitations)
    assert report.counts()["locations_invalid"] == 1


def test_stale_evidence_makes_the_level_insufficient(tmp_path):
    root = copy_sample(tmp_path / "repo")
    symbols = load_symbols(root)
    answer, snap = build(symbols, [(GOOD, ["E1"])], root=root)
    cart = root / "shop" / "cart.py"
    cart.write_text("# edited\n" + cart.read_text(encoding="utf-8"), encoding="utf-8")
    report = validate_answer(answer, snap, CONFIG, None)
    assert report.policy.level == LEVEL_INSUFFICIENT
    assert report.claims[0].final == UNSUPPORTED
    assert any("stale" in reason for reason in report.claims[0].reasons)
    assert any("valid citation" in text for text in report.policy.limitations)


def test_judge_failure_falls_back_to_rules(symbols):
    judge = ScriptedJudge(error=LLMConnectionError("Cannot reach Ollama"))
    report = run(symbols, [(GOOD, ["E1"])], judge)
    assert report.judge_error == "Cannot reach Ollama"
    assert (report.claims[0].final, report.claims[0].basis) == (SUPPORTED, "rules only")
    assert any("judge failed" in note for note in report.notes)


def test_unreadable_judge_output_is_reported(symbols):
    report = run(symbols, [(GOOD, ["E1"])], ScriptedJudge("I think it is fine"))
    assert report.claims[0].judge_verdict == JUDGE_UNPARSED
    assert any("no readable verdict" in note for note in report.notes)


def test_judge_can_be_switched_off(symbols):
    judge = ScriptedJudge("F1: SUPPORTED")
    config = dataclasses.replace(CONFIG, judge_enabled=False)
    report = run(symbols, [(GOOD, ["E1"])], judge, config)
    assert judge.prompts == []
    assert report.claims[0].judge_verdict == JUDGE_SKIPPED
    assert any("judge was off" in note for note in report.notes)


def test_only_the_first_claims_go_to_the_judge(symbols):
    facts = [(GOOD, ["E1"]), ("`checkout` returns `amount`", ["E1"])]
    judge = ScriptedJudge("F1: SUPPORTED | ok\nF2: SUPPORTED | ok")
    config = dataclasses.replace(CONFIG, judge_max_claims=1)
    report = run(symbols, facts, judge, config)
    assert report.claims[1].judge_verdict == JUDGE_SKIPPED
    assert "F2 (" not in judge.prompts[0]


def test_abstention_has_no_claims_and_no_level(symbols):
    answer = make_answer(status=STATUS_INSUFFICIENT)
    snap = snapshot_for(SAMPLE_REPO, symbols)
    report = validate_answer(answer, snap, CONFIG, None)
    assert report.claims == []
    assert report.policy.level == LEVEL_NOT_APPLICABLE


def test_evidence_rows_show_scores_and_citation_status(symbols):
    answer, snap = build(symbols, [(GOOD, ["E1"])])
    report = validate_answer(answer, snap, CONFIG, None)
    (row,) = evidence_rows(answer, report, CONFIG)
    assert row["citation"] == CITE_VALID
    assert row["strong"] is True
    assert row["cited_by"] == "F1"
    assert row["similarity"] == 0.55


def test_candidate_rows_show_what_was_sent_to_the_model(symbols):
    answer, _ = build(symbols, [(GOOD, ["E1"])])
    sym = symbols["ShoppingCart.checkout"]
    other = symbols["ShoppingCart.total"]
    results = [
        SearchResult(sym, 0.03, 1, 0.55, 1, 12.0),
        SearchResult(other, 0.02, 2, 0.1, None, None),
    ]
    rows = candidate_rows(results, answer, QAConfig())
    assert [r["sent_as"] for r in rows] == ["E1", "-"]
    assert [r["passed_gate"] for r in rows] == [True, False]