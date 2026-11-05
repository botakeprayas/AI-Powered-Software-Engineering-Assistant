"""Evaluation log (JSONL): one line per question, so correctness can be counted later.

Code is NOT stored by default (only a hash and a length), because logs can leak source code.
The file is git-ignored. `human_label` stays null until you label records by hand.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from core.chat.models import Answer
from core.semantic.models import SearchResult
from core.validation.config import ValidationConfig
from core.validation.models import ValidationReport

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
THRESHOLD_FIELDS = (
    "min_valid_citations", "min_strong_evidence", "min_cosine", "min_keyword_score",
    "min_supported_fraction", "max_unsupported_facts", "fact_anchor_coverage",
    "inference_anchor_coverage", "word_coverage", "judge_enabled",
)
COUNT_KEYS = (
    "evidence_cited", "evidence_cited_valid", "locations_written", "locations_invalid",
    "locations_ungrounded", "claims_total", "claims_supported", "claims_weak",
    "claims_unsupported", "claims_unchecked",
)


def build_record(
    answer: Answer,
    report: ValidationReport,
    results: Sequence[SearchResult],
    config: ValidationConfig,
    repo_name: str,
    judge_name: str,
) -> dict:
    sent = {e.symbol.key: e.id for e in answer.evidence}
    retrieved = []
    for rank, result in enumerate(results, start=1):
        sym = result.symbol
        entry = {
            "rank": rank,
            "key": sym.key,
            "location": sym.location,
            "kind": sym.kind,
            "qualname": sym.qualname,
            "semantic_score": result.semantic_score,
            "keyword_rank": result.keyword_rank,
            "keyword_score": result.keyword_score,
            "sent_as": sent.get(sym.key),
            "code_sha256": hashlib.sha256(sym.code.encode("utf-8")).hexdigest(),
            "code_chars": len(sym.code),
        }
        if config.log_code:
            entry["code"] = sym.code
        retrieved.append(entry)
    policy = report.policy
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": answer.question,
        "repo": repo_name,
        "llm": {
            "provider": answer.llm_provider,
            "model": answer.llm_model,
            "is_mock": answer.is_mock,
        },
        "judge": {"model": report.judge_model or judge_name, "error": report.judge_error},
        "status": answer.status,
        "abstained_before_llm": answer.abstained_before_llm,
        "retrieved": retrieved,
        "evidence": [
            {
                "id": c.evidence_id,
                "location": c.location,
                "cited_by": report.cited_by.get(c.evidence_id, []),
                "status": c.status,
                "detail": c.detail,
            }
            for c in report.citations
        ],
        "claims": [
            {
                "label": c.label,
                "kind": c.kind,
                "text": c.text,
                "evidence_ids": list(c.evidence_ids),
                "citation_ok": c.citation_ok,
                "rule_verdict": c.rule_verdict,
                "rule_score": c.rule_score,
                "missing_anchors": list(c.missing_anchors),
                "judge_verdict": c.judge_verdict,
                "judge_reason": c.judge_reason,
                "final": c.final,
                "basis": c.basis,
                "reasons": list(c.reasons),
            }
            for c in report.claims
        ],
        "locations": [
            {"claim": x.claim_label, "text": x.text, "status": x.status, "detail": x.detail}
            for x in report.locations
        ],
        "answer": {
            "message": answer.message,
            "unknowns": list(answer.unknowns),
            "warnings": list(answer.warnings),
        },
        "policy": {
            "level": policy.level if policy else "not_validated",
            "limitations": list(policy.limitations) if policy else [],
            "checks": (
                [{"name": name, "passed": ok} for name, ok, _ in policy.checks] if policy else []
            ),
        },
        "counts": report.counts(),
        "notes": list(report.notes),
        "thresholds": {name: getattr(config, name) for name in THRESHOLD_FIELDS},
        "human_label": None,
    }


def append_record(path: Path, record: dict) -> bool:
    """Append one JSON line. Never raises: a log problem must not break the app."""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not write the evaluation log: %s", exc)
        return False
    return True


def read_log(path: Path) -> tuple[list[dict], int]:
    """Return (records, number of malformed lines skipped). A missing file is empty."""
    records: list[dict] = []
    bad = 0
    target = Path(path)
    if not target.exists():
        return records, bad
    with target.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                bad += 1
    return records, bad


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def _rate(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole


def summarize(records: Iterable[dict]) -> dict:
    """Counts for citation correctness and unsupported claims. Automatic checks only."""
    totals: Counter[str] = Counter()
    status: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    count = 0
    for record in records:
        count += 1
        counts = record.get("counts") or {}
        for key in COUNT_KEYS:
            totals[key] += _as_int(counts.get(key, 0))
        status[str(record.get("status", "unknown"))] += 1
        levels[str((record.get("policy") or {}).get("level", "unknown"))] += 1
    return {
        "records": count,
        "status": dict(status),
        "levels": dict(levels),
        "totals": dict(totals),
        "citation_valid_rate": _rate(totals["evidence_cited_valid"], totals["evidence_cited"]),
        "claim_supported_rate": _rate(totals["claims_supported"], totals["claims_total"]),
        "claim_unsupported_rate": _rate(totals["claims_unsupported"], totals["claims_total"]),
    }


def _pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{part} of {whole} ({100 * part / whole:.1f}%)"


def format_summary(summary: dict, bad_lines: int = 0) -> str:
    t = summary["totals"]
    claims = t.get("claims_total", 0)
    status = ", ".join(f"{k}={v}" for k, v in sorted(summary["status"].items())) or "none"
    levels = ", ".join(f"{k}={v}" for k, v in sorted(summary["levels"].items())) or "none"
    lines = [
        "Evaluation log summary",
        f"Records: {summary['records']} (malformed lines skipped: {bad_lines})",
        f"Answer status: {status}",
        f"Evidence level: {levels}",
        f"Citations valid: {_pct(t.get('evidence_cited_valid', 0), t.get('evidence_cited', 0))}",
        f"Claims supported: {_pct(t.get('claims_supported', 0), claims)}",
        f"Claims weak: {t.get('claims_weak', 0)}",
        f"Claims unsupported: {_pct(t.get('claims_unsupported', 0), claims)}",
        f"Locations invented by the model: {t.get('locations_invalid', 0)}",
        "These counts come from automatic checks. They are not human-verified correctness.",
    ]
    return "\n".join(lines)