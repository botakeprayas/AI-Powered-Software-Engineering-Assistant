"""Turn an Answer into Markdown (used by the Streamlit page and the CLI)."""

from __future__ import annotations

from core.chat.models import (
    INSUFFICIENT_MESSAGE,
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    Answer,
    Claim,
    Evidence,
)

STATUS_LABELS = {
    STATUS_ANSWERED: "Answered",
    STATUS_PARTIAL: "Partially answered",
    STATUS_INSUFFICIENT: "Insufficient evidence",
    STATUS_ERROR: "Error",
}


def citation_text(evidence: Evidence) -> str:
    return f"`{evidence.location}`"  # e.g. `shop/cart.py:22-27`


def _claim_line(claim: Claim, by_id: dict[str, Evidence]) -> str:
    cites = " ".join(citation_text(by_id[i]) for i in claim.evidence_ids if i in by_id)
    return f"- {claim.text} {cites}".rstrip()


def answer_to_markdown(answer: Answer) -> str:
    parts: list[str] = []
    if answer.is_mock:
        parts.append("> **MOCK provider:** canned text, not an answer from a real model.")
    parts.append(f"**{STATUS_LABELS.get(answer.status, answer.status)}**")
    if answer.status == STATUS_ERROR:
        parts.append(answer.message)
        return "\n\n".join(parts)
    by_id = answer.evidence_by_id()
    if answer.status == STATUS_INSUFFICIENT:
        parts.append(answer.message or INSUFFICIENT_MESSAGE)
        if answer.unknowns:
            missing = "\n".join(f"- {u}" for u in answer.unknowns)
            parts.append(f"**What is missing**\n\n{missing}")
        return "\n\n".join(parts)
    if answer.facts:
        lines = "\n".join(_claim_line(c, by_id) for c in answer.facts)
        parts.append(f"**Facts from the code**\n\n{lines}")
    if answer.inferences:
        lines = "\n".join(_claim_line(c, by_id) for c in answer.inferences)
        parts.append(f"**Inferences (reasoned, not stated in the code)**\n\n{lines}")
    if answer.unknowns:
        lines = "\n".join(f"- {u}" for u in answer.unknowns)
        parts.append(f"**Unknowns**\n\n{lines}")
    return "\n\n".join(parts)