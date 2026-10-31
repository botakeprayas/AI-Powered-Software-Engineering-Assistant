"""ValidatingQAEngine: the Phase 3 QAEngine plus validation, an evidence panel and a log."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.chat.models import Answer, QAConfig, RepoInfo
from core.chat.qa import QAEngine
from core.semantic.models import SearchResult
from core.validation.config import ValidationConfig
from core.validation.evallog import append_record, build_record
from core.validation.models import ValidationReport
from core.validation.report import candidate_rows, evidence_rows, validate_answer
from core.validation.snapshot import IndexSnapshot
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RecordingRetriever:
    """Wraps the search engine and remembers what it returned (for the evidence panel)."""

    def __init__(self, retriever) -> None:
        self._retriever = retriever
        self.last: list[SearchResult] = []

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[SearchResult]:
        results = self._retriever.search(query, mode=mode, top_k=top_k)
        self.last = list(results)
        return results


@dataclass
class ValidatedAnswer:
    answer: Answer
    report: ValidationReport
    candidates: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    logged: bool = False
    validation_error: str = ""


class ValidatingQAEngine:
    def __init__(
        self,
        search_engine,
        llm: LLMService,
        root: Path,
        qa_config: QAConfig | None = None,
        config: ValidationConfig | None = None,
        repo: RepoInfo | None = None,
        judge: LLMService | None = None,
    ) -> None:
        """`search_engine` is a synced Phase 2 CodeSearchEngine; `root` is the repo folder."""
        self._search = search_engine
        self._root = Path(root)
        self._recorder = RecordingRetriever(search_engine)
        self._qa_config = qa_config or QAConfig()
        self._config = config or ValidationConfig()
        self._repo = repo or RepoInfo()
        self._judge = judge if judge is not None else llm
        self._qa = QAEngine(self._recorder, llm, self._qa_config, self._repo)

    def ask(self, question: str) -> ValidatedAnswer:
        """Answer, then validate. Never raises: failures become messages on the result."""
        self._recorder.last = []
        answer = self._qa.ask_safe(question)
        error = ""
        try:
            snapshot = IndexSnapshot.from_engine(self._root, self._search)
            report = validate_answer(answer, snapshot, self._config, self._judge)
        except Exception:
            logger.exception("Answer validation failed")
            error = (
                "Validation could not run, so this answer is NOT validated. "
                "Details are in the terminal log."
            )
            report = ValidationReport(notes=[error])
        result = ValidatedAnswer(
            answer=answer,
            report=report,
            candidates=candidate_rows(self._recorder.last, answer, self._qa_config),
            evidence=evidence_rows(answer, report, self._config),
            validation_error=error,
        )
        if self._config.log_enabled:
            record = build_record(
                answer, report, self._recorder.last, self._config,
                self._repo.name, self._judge.model_name,
            )
            result.logged = append_record(self._config.log_path, record)
        return result