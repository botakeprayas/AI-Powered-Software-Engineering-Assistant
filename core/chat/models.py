"""Data types for the repository chat."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.semantic.models import Symbol

STATUS_ANSWERED = "answered"
STATUS_PARTIAL = "partial"
STATUS_INSUFFICIENT = "insufficient"
STATUS_ERROR = "error"

INSUFFICIENT_MESSAGE = (
    "The retrieved code does not contain enough evidence to answer this question, "
    "so I will not guess."
)
NO_RELEVANT_CODE_MESSAGE = (
    "I could not find indexed code that is relevant to this question, so I will not guess."
)


@dataclass(frozen=True)
class QAConfig:
    candidate_k: int = 8  # results requested from hybrid search
    max_evidence: int = 5  # excerpts sent to the model
    min_cosine: float = 0.30  # semantic relevance threshold
    min_keyword_score: float = 8.0  # BM25 threshold (only for keyword rank 1-3)
    max_code_chars: int = 1500  # per excerpt
    max_evidence_chars: int = 8000  # all excerpts together
    max_question_chars: int = 1000
    max_answer_tokens: int = 800
    temperature: float = 0.1


@dataclass(frozen=True)
class RepoInfo:
    name: str = "repository"
    symbol_count: int = 0


@dataclass(frozen=True)
class Evidence:
    """One retrieved code excerpt, labeled E1, E2, ... for citation."""

    id: str
    symbol: Symbol
    code: str  # what the model saw (possibly truncated)
    code_truncated: bool
    semantic_score: float | None
    keyword_rank: int | None
    keyword_score: float | None

    @property
    def location(self) -> str:
        return self.symbol.location  # "file:start-end", straight from the index


@dataclass(frozen=True)
class Claim:
    text: str  # citations removed
    evidence_ids: tuple[str, ...]  # validated against the retrieved evidence


@dataclass
class Answer:
    question: str
    status: str
    facts: list[Claim] = field(default_factory=list)
    inferences: list[Claim] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    abstained_before_llm: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    is_mock: bool = False
    raw_text: str = ""

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}

    def citations(self) -> list[Evidence]:
        """Evidence actually cited by a fact or inference, in order of first use."""
        by_id = self.evidence_by_id()
        seen: set[str] = set()
        ordered: list[Evidence] = []
        for claim in [*self.facts, *self.inferences]:
            for eid in claim.evidence_ids:
                if eid in by_id and eid not in seen:
                    seen.add(eid)
                    ordered.append(by_id[eid])
        return ordered