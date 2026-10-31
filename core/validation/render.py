"""Plain-text rendering of validation results (used by the command-line tools)."""

from __future__ import annotations

from core.validation.models import (
    LEVEL_DISCLAIMER,
    LEVEL_NOT_APPLICABLE,
    SUPPORTED,
    UNCHECKED,
    UNSUPPORTED,
    WEAK,
)

FLAG_TEXT = {
    SUPPORTED: "supported",
    WEAK: "weakly supported",
    UNSUPPORTED: "UNSUPPORTED",
    UNCHECKED: "could not be checked",
}


def validation_to_text(validated) -> str:
    report = validated.report
    lines: list[str] = []
    if validated.validation_error:
        lines.append(f"NOT VALIDATED: {validated.validation_error}")
    policy = report.policy
    if policy is not None and policy.level != LEVEL_NOT_APPLICABLE:
        lines.append(f"Evidence level: {policy.level} (rule-based)")
        lines.append(LEVEL_DISCLAIMER)
        lines += [f"  LIMITATION: {text}" for text in policy.limitations]
    for check in report.claims:
        flag = FLAG_TEXT.get(check.final, check.final)
        lines.append(f"  {check.label} [{check.kind}] {flag} ({check.basis}): {check.text}")
        lines += [f"      - {reason}" for reason in check.reasons]
        if check.judge_reason:
            lines.append(f"      - judge: {check.judge_reason}")
    for loc in report.locations:
        where = f"{loc.text} in {loc.claim_label}"
        lines.append(f"  location {where}: {loc.status} ({loc.detail})")
    lines += [f"  note: {note}" for note in report.notes]
    return "\n".join(lines) if lines else "No validation details."