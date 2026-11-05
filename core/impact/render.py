"""Plain-text and table rendering of a change-impact report.

Used by the command line and the Streamlit page. Scores are shown as heuristic ranks with
the points of every factor, never as probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.impact.analyze import (
    AffectedFile,
    Comparison,
    ImpactReport,
    RecommendedTest,
    UnitImpact,
)
from core.impact.changes import ChangedUnit
from core.impact.rank import Factor
from core.impact.traverse import Dependent


def unit_location(unit: ChangedUnit) -> str:
    if unit.is_module_level or not unit.start_line:
        return unit.file
    return f"{unit.file}:{unit.start_line}-{unit.end_line}"


def factor_line(factor: Factor) -> str:
    if not factor.available:
        return f"{factor.name}: not computed (left out of the score) | {factor.value}"
    return f"{factor.name}: {factor.points}/{factor.max_points} points | {factor.value}"


def factor_rows(impact: UnitImpact) -> list[dict]:
    rows = []
    for factor in impact.factors:
        if factor.available:
            points = f"{factor.points}/{factor.max_points}"
        else:
            points = "not computed"
        rows.append(
            {
                "factor": factor.name,
                "points": points,
                "value": factor.value,
                "what it means": factor.explanation,
            }
        )
    return rows


def unit_rows(report: ImpactReport) -> list[dict]:
    return [
        {
            "rank": item.rank,
            "changed code": item.unit.qualname,
            "kind": item.unit.kind,
            "where": unit_location(item.unit),
            "heuristic rank (0-100)": item.score,
            "band": item.band,
            "known dependents": len(item.known),
            "inferred dependents": len(item.inferred),
            "tests reaching it": len(item.traversal.known_tests),
        }
        for item in report.units
    ]


def dependent_rows(deps: Sequence[Dependent]) -> list[dict]:
    return [
        {
            "code": dep.label,
            "file": dep.file,
            "kind": dep.kind,
            "depth": dep.depth,
            "first link": dep.location,
            "why": dep.explain(),
        }
        for dep in deps
    ]


def file_rows(files: Sequence[AffectedFile]) -> list[dict]:
    return [
        {
            "file": item.path,
            "depth": item.min_depth,
            "also edited by the diff": item.also_changed,
            "through": ", ".join(item.labels),
            "why": item.reasons[0] if item.reasons else "",
        }
        for item in files
    ]


def recommended_rows(tests: Sequence[RecommendedTest]) -> list[dict]:
    return [{"test": t.id, "status": t.status, "why": t.reason} for t in tests]


def skipped_rows(report: ImpactReport) -> list[dict]:
    return [
        {"file": path, "outcome": outcome, "detail": detail}
        for path, outcome, detail in report.skipped
    ]


def _dependents_lines(title: str, deps: Sequence[Dependent], limit: int = 10) -> list[str]:
    if not deps:
        return [f"   {title}: none"]
    lines = [f"   {title}:"]
    for dep in list(deps)[:limit]:
        where = f"{dep.file}, depth {dep.depth}"
        lines.append(f"     {dep.label} ({where}) - {dep.explain()}")
    if len(deps) > limit:
        lines.append(f"     ... and {len(deps) - limit} more")
    return lines


def _unit_block(impact: UnitImpact) -> list[str]:
    unit = impact.unit
    head = (
        f"{impact.rank}. {unit.qualname} [{unit.kind}] {unit_location(unit)} - "
        f"heuristic rank {impact.score} of 100 ({impact.band})"
    )
    lines = [head, f"   what changed: {unit.describe()}", "   factors:"]
    for factor in impact.factors:
        lines.append(f"     {factor_line(factor)}")
        lines.append(f"       {factor.explanation}")
    for finding in impact.findings:
        lines.append(f"   ruff {finding.code} at line {finding.line}: {finding.message}")
    lines += _dependents_lines("known dependents (confirmed)", impact.known)
    lines += _dependents_lines("inferred dependents (NOT confirmed)", impact.inferred)
    lines += [f"   note: {note}" for note in impact.notes]
    return lines


def _file_lines(files: Sequence[AffectedFile]) -> list[str]:
    if not files:
        return ["  (none found)"]
    lines = []
    for item in files:
        edited = " [also edited by the diff]" if item.also_changed else ""
        names = ", ".join(item.labels)
        lines.append(f"  {item.path} (depth {item.min_depth}){edited}: {names}")
        lines += [f"      why: {reason}" for reason in item.reasons]
    return lines


def _test_lines(report: ImpactReport) -> list[str]:
    if not report.tests:
        return [
            "  (none found: no test reaches the changed code through the graph, "
            "and none is named after a changed file)"
        ]
    lines = [f"  [{t.status}] {t.id} - {t.reason}" for t in report.tests]
    if report.test_command:
        lines.append(f"  run: {report.test_command}")
    return lines


def _comparison_lines(comparisons: Mapping[str, Comparison]) -> list[str]:
    lines = ["", "COMPARISON WITH YOUR HAND-CHECKED LISTS"]
    for name, comparison in comparisons.items():
        lines.append(f"  {name}: {comparison.summary()}")
        if comparison.missed:
            lines.append(f"    missed: {', '.join(comparison.missed)}")
        if comparison.found_inferred_only:
            lines.append(f"    found only as inferred: {', '.join(comparison.found_inferred_only)}")
        if comparison.extra_known:
            lines.append(f"    known but not on your list: {', '.join(comparison.extra_known)}")
        if comparison.extra_inferred:
            lines.append(
                f"    inferred but not on your list: {', '.join(comparison.extra_inferred)}"
            )
    return lines


def report_to_text(
    report: ImpactReport, comparisons: Mapping[str, Comparison] | None = None
) -> str:
    lines = ["CHANGE-IMPACT REPORT", report.disclaimer, ""]
    if report.units:
        lines.append("CHANGED CODE (heuristic ranks, highest first)")
        for impact in report.units:
            lines += _unit_block(impact) + [""]
    else:
        lines += ["No changed Python code could be mapped, so no impact was computed.", ""]
    lines.append("KNOWN DEPENDENCIES (confirmed edges only)")
    lines += _file_lines(report.affected_known)
    lines += ["", "INFERRED IMPACTS (at least one uncertain link; NOT confirmed)"]
    lines += _file_lines(report.affected_inferred)
    lines += ["", "RECOMMENDED TESTS"] + _test_lines(report)
    if report.skipped:
        lines += ["", "NOT ANALYZED"]
        lines += [f"  {p}: {d} ({o})" for p, o, d in report.skipped]
    if report.notes:
        lines += ["", "NOTES"] + [f"  - {note}" for note in report.notes]
    if comparisons:
        lines += _comparison_lines(comparisons)
    lines += ["", "WHAT THIS ANALYSIS CANNOT SEE"] + [f"  - {text}" for text in report.limitations]
    return "\n".join(lines)