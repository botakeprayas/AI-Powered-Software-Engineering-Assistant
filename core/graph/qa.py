"""GraphQAEngine: the Phase 3 QAEngine plus call-graph expansion of the retrieved code."""

from __future__ import annotations

import logging

from core.chat.models import Answer, Evidence, QAConfig, RepoInfo
from core.chat.parsing import build_answer
from core.chat.qa import QAEngine, Retriever, select_evidence
from core.graph.expand import ExpansionConfig, ExpansionResult, GraphContext, expand_evidence
from core.graph.prompt import build_graph_prompt
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class GraphQAEngine(QAEngine):
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMService,
        context: GraphContext,
        config: QAConfig | None = None,
        repo: RepoInfo | None = None,
        expansion: ExpansionConfig | None = None,
    ) -> None:
        super().__init__(retriever, llm, config, repo)
        self._context = context
        self._expansion = expansion or ExpansionConfig()
        self.last_expansion = ExpansionResult()  # what the most recent question added

    def ask(self, question: str) -> Answer:
        """Like QAEngine.ask, with confirmed callers/callees added. ask_safe() is inherited."""
        question = self._clean_question(question)
        results = self._retriever.search(question, mode="hybrid", top_k=self._cfg.candidate_k)
        seeds = select_evidence(results, self._cfg)
        if not seeds:  # the graph cannot rescue an irrelevant question
            self.last_expansion = ExpansionResult()
            return self._abstain_before_llm(question, results)
        warning = ""
        try:
            expansion = expand_evidence(self._context, seeds, self._cfg, self._expansion)
        except Exception:
            logger.exception("Graph expansion failed")
            expansion = ExpansionResult()
            warning = "Graph expansion failed, so this answer uses search results only."
        self.last_expansion = expansion
        evidence: list[Evidence] = [*seeds, *expansion.evidence]
        prompt = build_graph_prompt(question, evidence, self._repo, expansion)
        response = self._llm.generate(
            prompt.user,
            system=prompt.system,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_answer_tokens,
        )
        answer = build_answer(question, response, evidence)
        if warning:
            answer.warnings.append(warning)
        return answer