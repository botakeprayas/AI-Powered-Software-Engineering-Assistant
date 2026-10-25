"""Turn the model's text into validated claims. Nothing here trusts the model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.chat.models import (
    INSUFFICIENT_MESSAGE,
    STATUS_ANSWERED,
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    Answer,
    Claim,
    Evidence,
)
from services.llm_service import LLMResponse

_CITE_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.IGNORECASE)
_HEADER_COLON = re.compile(
    r"^[\s#*>]*(STATUS|FACTS|INFERENCES|UNKNOWNS)[\s*]*:[\s*]*(.*)$", re.IGNORECASE
)
_HEADER_ALONE = re.compile(r"^[\s#*>]*(FACTS|INFERENCES|UNKNOWNS)[\s*]*$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)$")
_EMPTY_ITEMS = frozenset({"", "none", "n/a", "na", "nothing"})


def extract_ids(text: str) -> list[str]:
    """Evidence IDs cited in `text`, upper-cased, in order, without repeats."""
    found: list[str] = []
    for match in _CITE_GROUP.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            eid = part.strip().upper()
            if eid and eid not in found:
                found.append(eid)
    return found


def strip_citations(text: str) -> str:
    cleaned = _CITE_GROUP.sub("", text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


@dataclass
class ParsedOutput:
    status: str | None = None  # answered | partial | insufficient | None (not stated)
    facts: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    recognized: bool = False  # at least one required heading was found


def _parse_status(value: str) -> str | None:
    upper = value.upper()
    if "INSUFFICIENT" in upper:
        return STATUS_INSUFFICIENT
    if "PARTIAL" in upper:
        return STATUS_PARTIAL
    if "ANSWER" in upper:
        return STATUS_ANSWERED
    return None


def parse_model_output(text: str) -> ParsedOutput:
    parsed = ParsedOutput()
    buckets = {
        "FACTS": parsed.facts,
        "INFERENCES": parsed.inferences,
        "UNKNOWNS": parsed.unknowns,
    }
    bucket: list[str] | None = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        header = _HEADER_COLON.match(raw) or _HEADER_ALONE.match(raw)
        if header:
            parsed.recognized = True
            name = header.group(1).upper()
            rest = header.group(2).strip() if header.re.groups >= 2 else ""
            if name == "STATUS":
                parsed.status = _parse_status(rest)
                bucket = None
                continue
            bucket = buckets[name]
            if rest:
                bullet = _BULLET.match(rest)
                bucket.append(bullet.group(1).strip() if bullet else rest)
            continue
        if bucket is None:
            continue
        bullet = _BULLET.match(raw)
        if bullet:
            bucket.append(bullet.group(1).strip())
        elif raw[:1].isspace() and bucket:
            bucket[-1] = f"{bucket[-1]} {raw.strip()}"  # wrapped continuation line
        else:
            bucket.append(raw.strip())
    for items in buckets.values():
        items[:] = [i for i in items if re.sub(r"[^\w/]+", "", i.lower()) not in _EMPTY_ITEMS]
    return parsed


def _validate_claims(
    raw_items: list[str], valid_ids: set[str], label: str
) -> tuple[list[Claim], list[str], int]:
    claims: list[Claim] = []
    warnings: list[str] = []
    dropped = 0
    for raw in raw_items:
        ids = extract_ids(raw)
        good = tuple(i for i in ids if i in valid_ids)
        bad = [i for i in ids if i not in valid_ids]
        text = strip_citations(raw)
        if bad:
            warnings.append(f"Removed unknown evidence ID(s) {', '.join(bad)} from a {label}.")
        if not good or not text:
            dropped += 1
            continue
        claims.append(Claim(text=text, evidence_ids=good))
    if dropped:
        warnings.append(f"Dropped {dropped} {label}(s) without a valid citation.")
    return claims, warnings, dropped


def build_answer(question: str, response: LLMResponse, evidence: list[Evidence]) -> Answer:
    """Validate the model's text against the evidence it was given."""
    answer = Answer(
        question=question,
        status=STATUS_INSUFFICIENT,
        evidence=list(evidence),
        llm_provider=response.provider,
        llm_model=response.model,
        is_mock=response.is_mock,
        raw_text=response.text,
    )
    if response.truncated:
        answer.warnings.append("The model's answer hit its length limit and may be incomplete.")
    parsed = parse_model_output(response.text)
    if not parsed.recognized:
        answer.warnings.append(
            "The model did not follow the required answer format, so its text was not "
            "accepted as an answer."
        )
        answer.message = INSUFFICIENT_MESSAGE
        return answer
    answer.unknowns = [u for u in (strip_citations(x) for x in parsed.unknowns) if u]
    if parsed.status == STATUS_INSUFFICIENT:
        answer.message = INSUFFICIENT_MESSAGE
        return answer

    valid = {e.id for e in evidence}
    facts, fact_warnings, fact_drops = _validate_claims(parsed.facts, valid, "fact")
    inferences, inf_warnings, inf_drops = _validate_claims(
        parsed.inferences, valid, "inference"
    )
    answer.warnings += fact_warnings + inf_warnings
    if not facts and not inferences:
        answer.warnings.append("No statement had a valid citation, so no answer is shown.")
        answer.message = INSUFFICIENT_MESSAGE
        return answer
    answer.facts, answer.inferences = facts, inferences
    status = parsed.status or STATUS_ANSWERED
    if (fact_drops or inf_drops) and status == STATUS_ANSWERED:
        status = STATUS_PARTIAL
    answer.status = status
    return answer