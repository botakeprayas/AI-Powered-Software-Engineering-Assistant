"""Question answering workflow: retrieve -> gate -> prompt -> LLM -> validate."""

from __future__ import annotations

import logging
from typing import Protocol

from core.chat.models import (
    NO_RELEVANT_CODE_MESSAGE,
    STATUS_ERROR,
    STATUS_INSUFFICIENT,
    Answer,
    Evidence,
    QAConfig,
    RepoInfo,
)
from core.chat.parsing import build_answer
from core.chat.prompt import build_prompt
from core.semantic.models import SearchResult
from services.embedding_service import EmbeddingError
from services.llm_service import LLMError, LLMService

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "\n[... truncated ...]"


class Retriever(Protocol):
    """CodeSearchEngine from Phase 2 satisfies this."""

    def search(
        self, query: str, mode: str = "hybrid", top_k: int = 5
    ) -> list[SearchResult]: ...


def is_relevant(result: SearchResult, config: QAConfig) -> bool:
    """Evidence gate: a decent semantic match, or a strong match on exact keywords."""
    if result.semantic_score is not None and result.semantic_score >= config.min_cosine:
        return True
    return (
        result.keyword_rank is not None
        and result.keyword_rank <= 3
        and result.keyword_score is not None
        and result.keyword_score >= config.min_keyword_score
    )


def select_evidence(results: list[SearchResult], config: QAConfig) -> list[Evidence]:
    evidence: list[Evidence] = []
    used = 0
    for result in results:
        if len(evidence) >= config.max_evidence:
            break
        if not is_relevant(result, config):
            continue
        code = result.symbol.code
        truncated = len(code) > config.max_code_chars
        if truncated:
            code = code[: config.max_code_chars].rstrip() + TRUNCATION_MARKER
        if evidence and used + len(code) > config.max_evidence_chars:
            break
        used += len(code)
        evidence.append(
            Evidence(
                id=f"E{len(evidence) + 1}",
                symbol=result.symbol,
                code=code,
                code_truncated=truncated,
                semantic_score=result.semantic_score,
                keyword_rank=result.keyword_rank,
                keyword_score=result.keyword_score,
            )
        )
    return evidence


class QAEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMService,
        config: QAConfig | None = None,
        repo: RepoInfo | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._cfg = config or QAConfig()
        self._repo = repo or RepoInfo()

    def _clean_question(self, question: str) -> str:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Please type a question.")
        limit = self._cfg.max_question_chars
        if len(cleaned) > limit:
            raise ValueError(f"The question is too long (limit {limit} characters).")
        return cleaned

    def ask(self, question: str) -> Answer:
        """Answer a question. Raises LLMError (and friends) on failure; see ask_safe()."""
        question = self._clean_question(question)
        results = self._retriever.search(question, mode="hybrid", top_k=self._cfg.candidate_k)
        evidence = select_evidence(results, self._cfg)
        if not evidence:
            return self._abstain_before_llm(question, results)
        prompt = build_prompt(question, evidence, self._repo)
        response = self._llm.generate(
            prompt.user,
            system=prompt.system,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_answer_tokens,
        )
        return build_answer(question, response, evidence)

    def ask_safe(self, question: str) -> Answer:
        """Like ask(), but never raises: problems become an Answer with status 'error'."""
        try:
            return self.ask(question)
        except LLMError as exc:
            logger.warning("LLM error: %s (%s)", exc, type(exc).__name__)
            return self._error_answer(question, str(exc))
        except (ValueError, EmbeddingError) as exc:
            return self._error_answer(question, str(exc))
        except Exception:
            logger.exception("Unexpected error while answering")
            return self._error_answer(
                question, "Something went wrong while answering. Details are in the terminal log."
            )

    def _abstain_before_llm(self, question: str, results: list[SearchResult]) -> Answer:
        message = NO_RELEVANT_CODE_MESSAGE
        scores = [r.semantic_score for r in results if r.semantic_score is not None]
        if scores:
            message += (
                f" (Best semantic match scored {max(scores):.2f}; "
                f"the minimum is {self._cfg.min_cosine:.2f}.)"
            )
        return Answer(
            question=question,
            status=STATUS_INSUFFICIENT,
            message=message,
            abstained_before_llm=True,
            llm_provider=self._llm.provider,
            llm_model=self._llm.model_name,
            is_mock=self._llm.is_mock,
        )

    def _error_answer(self, question: str, message: str) -> Answer:
        return Answer(
            question=question,
            status=STATUS_ERROR,
            message=message,
            llm_provider=self._llm.provider,
            llm_model=self._llm.model_name,
            is_mock=self._llm.is_mock,
        )