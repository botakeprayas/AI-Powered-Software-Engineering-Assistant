"""Plain-text rendering of an investigation (used by the command line)."""

from __future__ import annotations

from dataclasses import dataclass

from core.bugs.context import BugEvidence
from core.bugs.investigate import Investigation
from core.bugs.proposal import DRAFT_LABEL, PROPOSED_LABEL
from core.chat.models import Evidence
from core.chat.render import answer_to_markdown
from core.graph.expand import GraphEvidence
from core.validation.models import ValidationReport
from core.validation.render import validation_to_text


@dataclass
class _Validated:
    report: ValidationReport
    validation_error: str = ""


def describe_evidence(item: Evidence) -> str:
    if isinstance(item, GraphEvidence):
        return f"via call graph: {item.via}"
    if isinstance(item, BugEvidence):
        return f"{item.role}: {item.why}"
    return "retrieved"


def investigation_to_text(inv: Investigation) -> str:
    parsed = inv.traceback
    exception = f"{parsed.exception_type or '?'}: {parsed.exception_message}"
    lines = [f"Exception: {exception}", "", "Frames:"]
    for number, item in enumerate(inv.mapped, start=1):
        where = f"{item.symbol.qualname} ({item.status})" if item.symbol else item.reason
        lines.append(f"  {number}. {item.location} in {item.frame.func or '?'} -> {where}")
    lines += ["", "Evidence:"]
    for ev in inv.context.evidence:
        lines.append(f"  {ev.id} {ev.symbol.qualname} {ev.location} [{describe_evidence(ev)}]")
    if not inv.context.evidence:
        lines.append("  (none)")
    for note in inv.context.expansion.notes:
        lines.append(f"  graph note: {note}")
    if inv.hypothesis is not None:
        lines += ["", "ROOT-CAUSE HYPOTHESIS (a hypothesis, not a fix)", ""]
        lines.append(answer_to_markdown(inv.hypothesis))
        if inv.report is not None:
            lines += ["", "Validation:", validation_to_text(_Validated(inv.report))]
    if inv.fix is not None:
        fix = inv.fix
        lines += ["", f"FIX [{fix.label}] status: {fix.status}"]
        if fix.rationale:
            lines.append(f"Rationale: {fix.rationale}")
        if fix.diff:
            lines += [f"Lines changed: {fix.changed_lines}", fix.diff.rstrip()]
        lines += [f"  problem: {text}" for text in fix.problems]
    if inv.test_draft is not None:
        draft = inv.test_draft
        lines += ["", f"REGRESSION TEST [{draft.label}] status: {draft.status}"]
        if draft.path:
            lines.append(f"Would be written to: {draft.path}")
        if draft.reason:
            lines.append(f"Reason: {draft.reason}")
        if draft.code:
            lines.append(draft.code.rstrip())
        lines += [f"  problem: {text}" for text in draft.problems]
    lines += ["", "Steps:"] + [f"  - {text}" for text in inv.steps]
    lines += [f"  ERROR: {text}" for text in inv.errors]
    lines.append(f"\nFix: {PROPOSED_LABEL}. Test: {DRAFT_LABEL}.")
    return "\n".join(lines)