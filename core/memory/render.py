"""Plain-text and table rendering for commit-aware memory (Phase 8)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.chat.models import INSUFFICIENT_MESSAGE, STATUS_INSUFFICIENT
from core.chat.render import STATUS_LABELS, answer_to_markdown
from core.memory.answers import AnswerRecord, VersionStamp
from core.memory.freshness import Freshness

if TYPE_CHECKING:
    from core.memory.service import VersionedAnswer


def stamp_text(stamp: VersionStamp | None) -> str:
    if stamp is None:
        return "no indexed version"
    branch = f", branch {stamp.branch}" if stamp.branch else ""
    return f"version {stamp.label} (indexed {stamp.indexed_at}{branch})"


def freshness_lines(freshness: Freshness) -> list[str]:
    lines = [f"Index status: {freshness.status.upper()}", freshness.headline]
    lines += [f"  - {reason}" for reason in freshness.reasons]
    return lines


def citation_rows(citations: Sequence[dict]) -> list[dict]:
    return [
        {
            "citation": c["id"],
            "where": c["location"],
            "symbol": c["symbol"],
            "version": c["version"],
            "commit": (c["commit"] or "-")[:12],
            "valid": "-" if c.get("valid") is None else c["valid"],
        }
        for c in citations
    ]


def history_rows(records: Sequence[AnswerRecord]) -> list[dict]:
    return [
        {
            "id": r.id,
            "asked at": r.asked_at,
            "question": r.question,
            "status": r.status,
            "version": r.version.label,
            "commit": (r.version.commit_hash or "-")[:12],
            "state": "OUTDATED" if r.outdated else "valid",
            "index was": r.freshness,
        }
        for r in records
    ]


def history_lines(records: Sequence[AnswerRecord]) -> list[str]:
    lines = []
    for r in records:
        state = "OUTDATED" if r.outdated else "valid"
        question = r.question if len(r.question) <= 60 else r.question[:57] + "..."
        lines.append(f"#{r.id} {r.asked_at} [{state}] {r.status} | {r.version.label} | {question}")
    return lines or ["No stored answers."]


def record_markdown(record: AnswerRecord) -> str:
    data = record.answer
    parts = []
    if record.outdated:
        parts.append(f"> **OUTDATED:** {record.outdated_reason}")
    if data.get("is_mock"):
        parts.append("> **MOCK provider:** canned text, not an answer from a real model.")
    parts.append(f"**{STATUS_LABELS.get(record.status, record.status)}**")
    where = {c["id"]: c["location"] for c in record.citations}

    def line(claim: dict) -> str:
        spots = " ".join(f"`{where[i]}`" for i in claim["evidence_ids"] if i in where)
        return f"- {claim['text']} {spots}".rstrip()

    if record.status == STATUS_INSUFFICIENT:
        parts.append(data.get("message") or INSUFFICIENT_MESSAGE)
    else:
        if data.get("facts"):
            parts.append("**Facts from the code**\n\n" + "\n".join(map(line, data["facts"])))
        if data.get("inferences"):
            parts.append("**Inferences**\n\n" + "\n".join(map(line, data["inferences"])))
    if data.get("unknowns"):
        parts.append("**Unknowns**\n\n" + "\n".join(f"- {u}" for u in data["unknowns"]))
    return "\n\n".join(parts)


def versioned_text(answer: VersionedAnswer) -> str:
    lines: list[str] = []
    if answer.freshness is not None:
        lines += freshness_lines(answer.freshness)
    if answer.refreshed is not None:
        lines += ["", "Index update:", answer.refreshed.describe()]
    lines.append("")
    if answer.blocked:
        return "\n".join([*lines, f"BLOCKED: {answer.reason}"])
    if answer.cached is not None:
        lines.append(
            f"Reused stored answer #{answer.cached.id} (same indexed content and settings)."
        )
        lines.append(record_markdown(answer.cached))
    elif answer.validated is not None:
        lines.append(answer_to_markdown(answer.validated.answer))
    stamp = answer.stamp
    lines += ["", f"Based on {stamp_text(stamp)}"]
    if stamp is not None and stamp.commit_hash:
        dirty = " (plus uncommitted changes)" if stamp.dirty else ""
        lines.append(f"Commit: {stamp.commit_hash}{dirty}")
    lines.append("Citations:")
    for c in answer.citations:
        where = f"{c['id']} {c['location']} ({c['symbol']})"
        lines.append(f"  {where} @ {c['version']}; valid: {c['valid']}")
    if not answer.citations:
        lines.append("  (none)")
    if answer.record_id is not None:
        lines.append(f"History id: #{answer.record_id}")
    lines += [f"WARNING: {warning}" for warning in answer.warnings]
    return "\n".join(lines)