from datetime import datetime

from validation_e2e_helpers import BrokenStore, StubSearch, config_for
from validation_helpers import ScriptedJudge

from core.chat.models import QAConfig, RepoInfo
from core.validation.engine import ValidatingQAEngine
from core.validation.evallog import read_log, summarize
from core.validation.models import (
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    SUPPORTED,
)
from services.llm_service import LLMConnectionError, MockLLM

QUESTION = "how is the invoice total computed?"
OPEN_GATE = QAConfig(min_cosine=-1.0)
JUDGE_OK = "F1: SUPPORTED | ok\nF2: SUPPORTED | ok\nI1: SUPPORTED | ok"
FAKE_ANSWER = (
    "STATUS: ANSWERED\nFACTS:\n- total is computed at billing.py:90-120 [E1]\n"
    "INFERENCES:\n- none\nUNKNOWNS:\n- none"
)


def make_engine(engine, repo_dir, tmp_path, llm=None, judge=None, **changes):
    return ValidatingQAEngine(
        engine,
        llm or MockLLM(),
        repo_dir,
        OPEN_GATE,
        config_for(tmp_path, **changes),
        RepoInfo("repo", 10),
        judge=judge,
    )


def test_grounded_answer_is_validated_end_to_end(engine, repo_dir, tmp_path):
    engine.sync()
    result = make_engine(engine, repo_dir, tmp_path, judge=ScriptedJudge(JUDGE_OK)).ask(QUESTION)
    assert result.answer.status == "answered"
    assert result.report.citations
    assert all(c.ok for c in result.report.citations)
    assert all(c.final == SUPPORTED for c in result.report.claims)
    assert result.evidence and result.candidates
    assert any(row["sent_as"] != "-" for row in result.candidates)


def test_invented_location_in_an_answer_is_flagged(engine, repo_dir, tmp_path):
    engine.sync()
    validating = make_engine(
        engine, repo_dir, tmp_path,
        llm=ScriptedJudge(FAKE_ANSWER),
        judge=ScriptedJudge("F1: SUPPORTED | ok"),
    )
    result = validating.ask(QUESTION)
    report = result.report
    assert report.claims[0].final == "unsupported"
    assert report.counts()["locations_invalid"] == 1
    assert report.policy.level == LEVEL_LIMITED
    assert any("billing.py:90-120" in text for text in report.policy.limitations)


def test_stale_citations_make_the_evidence_insufficient(engine, repo_dir, tmp_path):
    engine.sync()
    for path in repo_dir.glob("*.py"):
        path.write_text("# edited\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    result = make_engine(engine, repo_dir, tmp_path, judge=ScriptedJudge("")).ask(QUESTION)
    assert result.report.policy.level == LEVEL_INSUFFICIENT
    assert all(c.final == "unsupported" for c in result.report.claims)
    assert any("valid citation" in text for text in result.report.policy.limitations)


def test_abstention_has_nothing_to_validate(tmp_path):
    llm = MockLLM()
    validating = ValidatingQAEngine(
        StubSearch(), llm, tmp_path, config=config_for(tmp_path), judge=ScriptedJudge("")
    )
    result = validating.ask("what is the capital of France?")
    assert result.answer.status == "insufficient"
    assert result.answer.abstained_before_llm
    assert llm.calls == 0
    assert result.report.claims == []
    assert result.report.policy.level == LEVEL_NOT_APPLICABLE


def test_judge_failure_does_not_stop_validation(engine, repo_dir, tmp_path):
    engine.sync()
    judge = ScriptedJudge(error=LLMConnectionError("Cannot reach Ollama"))
    result = make_engine(engine, repo_dir, tmp_path, judge=judge).ask(QUESTION)
    assert "Cannot reach Ollama" in result.report.judge_error
    assert all(c.basis == "rules only" for c in result.report.claims)
    assert result.report.policy is not None


def test_log_record_is_written_and_countable(engine, repo_dir, tmp_path):
    engine.sync()
    config = config_for(tmp_path, log_enabled=True)
    validating = ValidatingQAEngine(
        engine, MockLLM(), repo_dir, OPEN_GATE, config, RepoInfo("repo", 10),
        judge=ScriptedJudge(JUDGE_OK),
    )
    result = validating.ask(QUESTION)
    records, bad = read_log(config.log_path)
    assert result.logged and bad == 0 and len(records) == 1
    record = records[0]
    assert record["question"] == QUESTION
    assert record["counts"]["claims_total"] == len(result.report.claims)
    assert record["human_label"] is None
    assert all("code" not in item and item["code_sha256"] for item in record["retrieved"])
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None
    summary = summarize(records)
    assert summary["records"] == 1
    assert summary["citation_valid_rate"] == 1.0


def test_code_is_logged_only_when_asked(engine, repo_dir, tmp_path):
    engine.sync()
    config = config_for(tmp_path, log_enabled=True, log_code=True)
    validating = ValidatingQAEngine(
        engine, MockLLM(), repo_dir, OPEN_GATE, config, RepoInfo("repo", 10),
        judge=ScriptedJudge(JUDGE_OK),
    )
    validating.ask(QUESTION)
    records, _ = read_log(config.log_path)
    assert all("code" in item for item in records[0]["retrieved"])


def test_an_unwritable_log_does_not_crash_the_app(engine, repo_dir, tmp_path):
    engine.sync()
    validating = make_engine(
        engine, repo_dir, tmp_path, judge=ScriptedJudge(JUDGE_OK),
        log_enabled=True, log_path=tmp_path,
    )
    result = validating.ask(QUESTION)
    assert result.answer.status == "answered"
    assert result.logged is False


def test_a_validation_crash_is_contained(tmp_path):
    validating = ValidatingQAEngine(
        BrokenStore(), MockLLM(), tmp_path, config=config_for(tmp_path)
    )
    result = validating.ask("anything")
    assert "NOT validated" in result.validation_error
    assert result.report.policy is None
    assert result.answer.status == "insufficient"