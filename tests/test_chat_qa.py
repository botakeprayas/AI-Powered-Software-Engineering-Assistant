import pytest
from chat_helpers import FakeRetriever, make_result, make_symbol, make_total_symbol

from core.chat.models import NO_RELEVANT_CODE_MESSAGE, QAConfig, RepoInfo
from core.chat.qa import QAEngine, is_relevant, select_evidence
from core.chat.render import answer_to_markdown
from services.llm_service import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    MockLLM,
)

QUESTION = "How does checkout work?"


def make_qa(llm=None, results=None, config=None):
    if results is None:
        results = [make_result(make_symbol()), make_result(make_total_symbol(), kw_rank=2)]
    llm = llm or MockLLM()
    return QAEngine(FakeRetriever(results), llm, config, RepoInfo("shop", 13)), llm


def test_answer_cites_retrieved_code_with_correct_location():
    qa, llm = make_qa()
    answer = qa.ask(QUESTION)
    assert answer.status == "answered"
    assert answer.is_mock and llm.calls == 1
    assert answer.facts and all(claim.evidence_ids for claim in answer.facts)
    assert answer.citations()[0].location == "shop/cart.py:22-27"
    assert "`shop/cart.py:22-27`" in answer_to_markdown(answer)
    assert "MOCK" in answer_to_markdown(answer)


def test_prompt_given_to_the_model_contains_the_evidence():
    qa, llm = make_qa()
    qa.ask(QUESTION)
    assert "[E1] shop/cart.py:22-27" in llm.prompts[0]
    assert QUESTION in llm.prompts[0]


def test_irrelevant_retrieval_abstains_without_calling_the_llm():
    weak = [make_result(make_symbol(), semantic=0.12, kw_rank=None)]
    qa, llm = make_qa(results=weak)
    answer = qa.ask("What is the capital of France?")
    assert answer.status == "insufficient"
    assert answer.abstained_before_llm
    assert answer.message.startswith(NO_RELEVANT_CODE_MESSAGE)
    assert "0.12" in answer.message
    assert llm.calls == 0


def test_no_retrieval_results_abstains():
    qa, llm = make_qa(results=[])
    answer = qa.ask(QUESTION)
    assert answer.status == "insufficient" and llm.calls == 0


def test_model_can_abstain():
    qa, _ = make_qa(llm=MockLLM("abstain"))
    answer = qa.ask(QUESTION)
    assert answer.status == "insufficient"
    assert answer.facts == []
    assert "will not guess" in answer_to_markdown(answer)


def test_uncited_and_bad_citation_answers_are_not_accepted():
    for behavior in ("uncited", "bad_citation"):
        answer = make_qa(llm=MockLLM(behavior))[0].ask(QUESTION)
        assert answer.status == "insufficient"
        assert answer.facts == []
        assert answer.warnings


def test_unstructured_answer_is_rejected():
    answer = make_qa(llm=MockLLM("unstructured"))[0].ask(QUESTION)
    assert answer.status == "insufficient"
    assert answer.raw_text


ERRORS = [
    LLMConnectionError("Cannot reach Ollama at http://localhost:11434. Is Ollama running?"),
    LLMTimeoutError("Ollama did not answer within 120 seconds."),
    LLMModelNotFoundError("Model 'x' is not installed in Ollama. Run: ollama pull x"),
    LLMServerError("Ollama had a temporary problem (HTTP 500)."),
    LLMResponseError("The model returned an empty answer."),
]


@pytest.mark.parametrize("error", ERRORS, ids=lambda e: type(e).__name__)
def test_llm_errors_become_error_answers_not_crashes(error):
    qa, _ = make_qa(llm=MockLLM(error=error))
    with pytest.raises(type(error)):
        qa.ask(QUESTION)
    answer = qa.ask_safe(QUESTION)
    assert answer.status == "error"
    assert answer.message == str(error)


def test_unexpected_errors_are_contained():
    class Broken:
        def search(self, query, mode="hybrid", top_k=5):
            raise RuntimeError("index exploded")

    answer = QAEngine(Broken(), MockLLM()).ask_safe(QUESTION)
    assert answer.status == "error"
    assert "terminal log" in answer.message


def test_empty_and_overlong_questions_are_rejected():
    qa, _ = make_qa()
    with pytest.raises(ValueError):
        qa.ask("   ")
    assert qa.ask_safe("   ").status == "error"
    assert qa.ask_safe("x" * 5000).status == "error"


def test_evidence_gate_rules():
    config = QAConfig()
    sym = make_symbol()
    assert not is_relevant(make_result(sym, semantic=0.12, kw_rank=2, kw_score=3.0), config)
    assert is_relevant(make_result(sym, semantic=0.45, kw_rank=None), config)
    assert is_relevant(make_result(sym, semantic=0.10, kw_rank=1, kw_score=15.0), config)
    assert not is_relevant(make_result(sym, semantic=0.10, kw_rank=5, kw_score=20.0), config)


def test_evidence_budget_and_truncation():
    results = [make_result(make_symbol(qualname=f"f{i}", code="x" * 5000)) for i in range(10)]
    evidence = select_evidence(results, QAConfig(max_code_chars=100))
    assert len(evidence) == 5
    assert evidence[0].code_truncated
    assert evidence[0].code.endswith("[... truncated ...]")
    assert [e.id for e in evidence] == ["E1", "E2", "E3", "E4", "E5"]


def test_citations_match_indexed_locations_with_the_real_search_engine(engine, repo_dir):
    engine.sync()
    qa = QAEngine(engine, MockLLM(), QAConfig(min_cosine=-1.0), RepoInfo("repo", 10))
    answer = qa.ask("how is the invoice total computed?")
    assert answer.status == "answered"
    indexed = {f"{file}:{start}-{end}" for file, start, end in engine.stored_locations().values()}
    cited = answer.citations()
    assert cited
    for item in cited:
        assert item.location in indexed
        lines = (repo_dir / item.symbol.file).read_text(encoding="utf-8").split("\n")
        start, end = item.symbol.start_line, item.symbol.end_line
        assert "\n".join(lines[start - 1 : end]) == item.symbol.code