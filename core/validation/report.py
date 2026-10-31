"""Tie the checks together: citations, claim support, judge and policy -> ValidationReport."""

from __future__ import annotations

from collections.abc import Sequence

from core.chat.models import Answer, QAConfig
from core.chat.qa import is_relevant
from core.semantic.models import SearchResult
from core.validation.citations import check_citations, labeled_claims
from core.validation.config import ValidationConfig
from core.validation.judge import JudgeItem, run_judge
from core.validation.models import (
    JUDGE_UNPARSED,
    LOC_INVALID,
    ClaimCheck,
    ValidationReport,
)
from core.validation.policy import evaluate_policy, is_strong
from core.validation.snapshot import IndexSnapshot
from core.validation.support import decide, rule_check, seen_text
from services.llm_service import LLMService


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def validate_answer(
    answer: Answer,
    snapshot: IndexSnapshot,
    config: ValidationConfig,
    judge_llm: LLMService | None = None,
) -> ValidationReport:
    """Check an Answer against the current index. The model's own claims are never trusted."""
    report = ValidationReport()
    labeled = labeled_claims(answer)
    if not labeled:
        report.notes.append("The answer makes no claims, so there is nothing to validate.")
        report.policy = evaluate_policy(answer, [], [], [], config, report.notes)
        return report
    citations, locations, cited_by = check_citations(answer, snapshot)
    report.citations, report.locations, report.cited_by = citations, locations, cited_by
    detail = {c.evidence_id: c for c in citations}
    by_id = answer.evidence_by_id()
    invented: dict[str, list[str]] = {}
    for loc in locations:
        if loc.status == LOC_INVALID:
            invented.setdefault(loc.claim_label, []).append(loc.text)

    judge_items: list[JudgeItem] = []
    for label, kind, claim in labeled:
        valid = tuple(by_id[i] for i in claim.evidence_ids if i in by_id and detail[i].ok)
        check = ClaimCheck(
            label=label,
            kind=kind,
            text=claim.text,
            evidence_ids=claim.evidence_ids,
            citation_ok=bool(valid) and label not in invented,
        )
        for eid in claim.evidence_ids:
            if not detail[eid].ok:
                check.reasons.append(f"{eid}: {detail[eid].status} ({detail[eid].detail})")
        for text in invented.get(label, []):
            check.reasons.append(f"invented location {text} (not in the indexed repository)")
        if valid:
            seen = "\n".join(seen_text(e) for e in valid)
            result = rule_check(claim.text, kind, seen, config)
            check.rule_verdict, check.rule_score = result.verdict, result.score
            check.anchors, check.missing_anchors = result.anchors, result.missing
            if result.missing:
                check.reasons.append("not found in the cited code: " + ", ".join(result.missing))
            if any(e.code_truncated for e in valid):
                check.reasons.append("the cited code was cut short in the prompt")
        if check.citation_ok:
            judge_items.append(JudgeItem(label, kind, claim.text, valid))
        report.claims.append(check)

    _apply_judge(report, judge_items, judge_llm, config)
    for check in report.claims:
        check.final, check.basis = decide(
            check.citation_ok, check.rule_verdict, check.judge_verdict
        )
    report.policy = evaluate_policy(
        answer, citations, locations, report.claims, config, report.notes
    )
    return report


def _apply_judge(
    report: ValidationReport,
    items: list[JudgeItem],
    judge_llm: LLMService | None,
    config: ValidationConfig,
) -> None:
    if not config.judge_enabled or judge_llm is None:
        report.notes.append("The LLM judge was off: claim checks used the rule-based method only.")
        return
    batch = items[: config.judge_max_claims]
    verdicts, error = run_judge(judge_llm, batch, config.judge_max_code_chars)
    report.judge_model = judge_llm.model_name
    report.judge_error = error
    if judge_llm.is_mock:
        report.notes.append("The judge is a MOCK provider, so its verdicts are not real.")
    if error:
        report.notes.append(f"The LLM judge failed ({error}): claim checks used rules only.")
        return
    if len(items) > len(batch):
        report.notes.append(f"Only the first {len(batch)} claims were sent to the judge.")
    judged = {item.label for item in batch}
    unread = 0
    for check in report.claims:
        if check.label not in judged:
            continue
        verdict = verdicts.get(check.label)
        if verdict is None:
            check.judge_verdict = JUDGE_UNPARSED
            unread += 1
        else:
            check.judge_verdict, check.judge_reason = verdict.verdict, verdict.reason
    if unread:
        report.notes.append(f"The judge gave no readable verdict for {unread} claim(s).")


def evidence_rows(
    answer: Answer, report: ValidationReport, config: ValidationConfig
) -> list[dict]:
    """One row per excerpt the model was shown, for the evidence inspection panel."""
    status = {c.evidence_id: c for c in report.citations}
    rows = []
    for ev in answer.evidence:
        check = status.get(ev.id)
        rows.append(
            {
                "id": ev.id,
                "location": ev.location,
                "symbol": ev.symbol.qualname,
                "kind": ev.symbol.kind,
                "similarity": _round(ev.semantic_score),
                "keyword_rank": ev.keyword_rank,
                "keyword_score": _round(ev.keyword_score),
                "strong": is_strong(ev, config),
                "cited_by": ", ".join(report.cited_by.get(ev.id, [])) or "-",
                "citation": check.status if check else "not cited",
                "detail": check.detail if check else "",
            }
        )
    return rows


def candidate_rows(
    results: Sequence[SearchResult], answer: Answer, qa_config: QAConfig
) -> list[dict]:
    """Everything hybrid search returned, including what the gate kept out of the prompt."""
    sent = {e.symbol.key: e.id for e in answer.evidence}
    rows = []
    for rank, result in enumerate(results, start=1):
        sym = result.symbol
        rows.append(
            {
                "rank": rank,
                "symbol": sym.qualname,
                "location": sym.location,
                "similarity": _round(result.semantic_score),
                "keyword_rank": result.keyword_rank,
                "keyword_score": _round(result.keyword_score),
                "passed_gate": is_relevant(result, qa_config),
                "sent_as": sent.get(sym.key, "-"),
            }
        )
    return rows