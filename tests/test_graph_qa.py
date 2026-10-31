import pytest
from chat_helpers import FakeRetriever, make_result
from graph_helpers import GRAPH_SAMPLE, key_of, sample_context

from core.chat.models import QAConfig, RepoInfo
from core.graph.expand import GraphContext, GraphEvidence
from core.graph.qa import GraphQAEngine
from services.llm_service import LLMConnectionError, MockLLM

QUESTION = "What does OrderService.checkout call, and where is that code?"


@pytest.fixture(scope="module")
def context():
    return sample_context()


def make_qa(context, results, llm=None):
    llm = llm or MockLLM()
    repo = RepoInfo("graph_sample", len(context.symbols))
    return GraphQAEngine(FakeRetriever(results), llm, context, QAConfig(), repo), llm


def checkout_result(context, **scores):
    return make_result(context.symbols[key_of(context, "OrderService.checkout")], **scores)


def test_a_cross_file_question_retrieves_the_connected_code(context):
    qa, llm = make_qa(context, [checkout_result(context)])
    answer = qa.ask(QUESTION)
    assert answer.status == "answered"
    where = {e.location for e in answer.evidence}
    assert {"orders/service.py:16-21", "utils/helpers.py:9-11", "main.py:7-12"} <= where
    prompt = llm.prompts[0]
    assert "via call graph" in prompt
    assert "def apply_tax" in prompt  # code from another file is in the prompt
    assert "CALL GRAPH NOTES" in prompt
    assert "UNCERTAIN" in prompt and "self.gateway.charge" in prompt


def test_citations_can_point_at_graph_added_code_and_match_the_files(context):
    qa, _ = make_qa(context, [checkout_result(context)])
    answer = qa.ask(QUESTION)
    cited = answer.citations()
    assert any(isinstance(item, GraphEvidence) for item in cited)
    for item in answer.evidence:
        sym = item.symbol
        lines = (GRAPH_SAMPLE / sym.file).read_text(encoding="utf-8").split("\n")
        assert "\n".join(lines[sym.start_line - 1 : sym.end_line]) == sym.code


def test_the_graph_cannot_rescue_an_irrelevant_question(context):
    weak = make_result(
        context.symbols[key_of(context, "OrderService.checkout")], semantic=0.1, kw_rank=None
    )
    qa, llm = make_qa(context, [weak])
    answer = qa.ask("what is the capital of France?")
    assert answer.status == "insufficient"
    assert answer.abstained_before_llm
    assert llm.calls == 0
    assert qa.last_expansion.evidence == []


def test_a_broken_graph_falls_back_to_search_results_only(context):
    class BrokenGraph:
        def has_node(self, node_id):
            raise RuntimeError("graph exploded")

    broken = GraphContext(BrokenGraph(), context.symbols, context.root)
    qa, _ = make_qa(broken, [checkout_result(context)])
    answer = qa.ask(QUESTION)
    assert answer.status == "answered"
    assert any("Graph expansion failed" in w for w in answer.warnings)
    assert not [e for e in answer.evidence if isinstance(e, GraphEvidence)]


def test_llm_errors_still_become_error_answers(context):
    error = LLMConnectionError("Cannot reach Ollama")
    qa, _ = make_qa(context, [checkout_result(context)], MockLLM(error=error))
    answer = qa.ask_safe(QUESTION)
    assert answer.status == "error"
    assert answer.message == "Cannot reach Ollama"