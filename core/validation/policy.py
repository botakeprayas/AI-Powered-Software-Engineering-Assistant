"""Evidence-sufficiency policy. Levels are rule outcomes, never probabilities."""

from __future__ import annotations

from collections.abc import Sequence

from core.chat.models import Answer, Evidence
from core.validation.config import ValidationConfig
from core.validation.models import (
    KIND_FACT,
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    LEVEL_SUFFICIENT,
    LOC_INVALID,
    LOC_UNGROUNDED,
    SUPPORTED,
    UNSUPPORTED,
    CitationCheck,
    ClaimCheck,
    LocationCheck,
    PolicyResult,
)


def is_strong(evidence: Evidence, config: ValidationConfig) -> bool:
    """Retrieval-score test: a decent semantic match, or a strong match on exact keywords."""
    if evidence.semantic_score is not None and evidence.semantic_score >= config.min_cosine:
        return True
    return (
        evidence.keyword_rank is not None
        and evidence.keyword_rank <= 3
        and evidence.keyword_score is not None
        and evidence.keyword_score >= config.min_keyword_score
    )


def evaluate_policy(
    answer: Answer,
    citations: Sequence[CitationCheck],
    locations: Sequence[LocationCheck],
    claims: Sequence[ClaimCheck],
    config: ValidationConfig,
    notes: Sequence[str] = (),
) -> PolicyResult:
    """Apply the configured minimums. Every failed rule becomes an explicit limitation."""
    if not claims:  # an abstention or an error: nothing was claimed, nothing to validate
        return PolicyResult(LEVEL_NOT_APPLICABLE, (), tuple(notes), ())
    by_id = answer.evidence_by_id()
    valid_ids = [c.evidence_id for c in citations if c.ok]
    broken = [c for c in citations if not c.ok]
    invalid = [x for x in locations if x.status == LOC_INVALID]
    loose = [x for x in locations if x.status == LOC_UNGROUNDED]
    strong = [i for i in valid_ids if i in by_id and is_strong(by_id[i], config)]
    supported = sum(c.final == SUPPORTED for c in claims)
    bad_facts = [c.label for c in claims if c.kind == KIND_FACT and c.final == UNSUPPORTED]
    need_valid = max(config.min_valid_citations, 1)

    broken_text = ", ".join(f"{c.evidence_id} ({c.status})" for c in broken)
    invalid_text = ", ".join(x.text for x in invalid)
    loose_text = ", ".join(x.text for x in loose)
    checks = [
        (
            "valid citations",
            len(valid_ids) >= need_valid,
            f"Only {len(valid_ids)} valid citation(s) back this answer "
            f"(policy minimum: {need_valid}).",
        ),
        (
            "citations match the index",
            not broken,
            f"{len(broken)} citation(s) do not match the indexed repository: {broken_text}.",
        ),
        (
            "no invented locations",
            not invalid,
            f"{len(invalid)} location(s) written in the answer do not exist: {invalid_text}.",
        ),
        (
            "locations come from the evidence",
            not loose,
            f"The answer wrote location(s) that were not retrieved evidence: {loose_text}.",
        ),
        (
            "strong retrieval evidence",
            len(strong) >= config.min_strong_evidence,
            f"Retrieval scores of the cited code are below the configured thresholds "
            f"(similarity {config.min_cosine:.2f}, keyword score {config.min_keyword_score:g}): "
            f"{len(strong)} strong item(s), policy minimum {config.min_strong_evidence}.",
        ),
        (
            "supported claims",
            supported / len(claims) >= config.min_supported_fraction,
            f"Only {supported} of {len(claims)} claim(s) are supported "
            f"(policy minimum: {config.min_supported_fraction:.0%} of claims).",
        ),
        (
            "unsupported facts",
            len(bad_facts) <= config.max_unsupported_facts,
            f"{len(bad_facts)} fact(s) are not supported by their cited code: "
            f"{', '.join(bad_facts)}.",
        ),
    ]
    failed = [text for _name, ok, text in checks if not ok]
    if not valid_ids:
        level = LEVEL_INSUFFICIENT
    elif failed:
        level = LEVEL_LIMITED
    else:
        level = LEVEL_SUFFICIENT
    return PolicyResult(level, tuple(failed), tuple(notes), tuple(checks))