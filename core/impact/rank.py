"""Transparent heuristic ranking factors for change impact.

Every factor shows its points, its maximum and a plain explanation. The points are judgment
calls, not fitted to real defect data, so the total is a HEURISTIC RANK, never a probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

HEURISTIC_DISCLAIMER = (
    "These are heuristic ranks, not probabilities. The factors and point values are "
    "judgment calls that have not been calibrated against real defects, so they can order "
    "changed code for review but cannot say how likely a problem is."
)
BAND_LOW = "low"
BAND_MEDIUM = "medium"
BAND_HIGH = "high"
MEDIUM_FROM = 34
HIGH_FROM = 67


@dataclass(frozen=True)
class Factor:
    name: str
    points: int
    max_points: int
    available: bool
    value: str
    explanation: str


def dependents_factor(known: int, inferred: int) -> Factor:
    points = min(30, 10 * known + 3 * inferred)
    return Factor(
        "Dependents", points, 30, True, f"{known} known, {inferred} inferred",
        "Code that calls or imports the changed code. Scored 10 points per known dependent "
        "and 3 per inferred one, up to 30. More dependents means more code that can break.",
    )


def position_factor(depth: int, reaches_module: bool, truncated: bool, max_depth: int) -> Factor:
    points = {0: 0, 1: 5, 2: 10}.get(depth, 15)
    if reaches_module and depth:
        points += 5
    value = f"depth {depth}" if depth else "no known ripple"
    if reaches_module and depth:
        value += ", reaches module-level code"
    if truncated:
        value += f" (trace stopped at depth {max_depth})"
    return Factor(
        "Graph position", min(points, 20), 20, True, value,
        "How far the known ripple reaches over confirmed edges (depth 1 = direct callers). "
        "5, 10 or 15 points for depth 1, 2 or 3 or more, plus 5 if it reaches module-level "
        "code that runs on import or at an entry point. Far ripples are easy to forget.",
    )


def test_factor(known_tests: int) -> Factor:
    points = {0: 20, 1: 10, 2: 5}.get(known_tests, 0)
    return Factor(
        "Test reach", points, 20, True, f"{known_tests} test(s) reach it",
        "Tests that reach this code through confirmed edges. This is static reachability, "
        "not measured coverage: a reaching test may not exercise the changed lines. "
        "20 points with none, 10 with one, 5 with two, 0 with three or more.",
    )


test_factor.__test__ = False  # a factor builder, not a test: stop pytest from collecting it


def churn_factor(commits: int | None, days: int) -> Factor:
    if commits is None:
        return Factor(
            "Code churn", 0, 10, False, "not available",
            "Git history was not available, so this factor is left out of the score.",
        )
    points = 0 if commits == 0 else 3 if commits <= 2 else 6 if commits <= 5 else 10
    return Factor(
        "Code churn", points, 10, True, f"{commits} commit(s) in {days} days",
        f"Commits that touched this file in the last {days} days (per file, not per function). "
        "0 points for none, 3 for 1-2, 6 for 3-5, 10 for 6 or more. Frequently edited files "
        "tend to be less stable.",
    )


def lint_factor(codes: Sequence[str] | None) -> Factor:
    if codes is None:
        return Factor(
            "Static analysis", 0, 10, False, "not available",
            "Ruff was not available, so this factor is left out of the score.",
        )
    count = len(codes)
    points = 0 if count == 0 else 5 if count == 1 else 10
    shown = ", ".join(sorted(set(codes))[:5])
    value = f"{count} finding(s)" + (f": {shown}" if shown else "")
    return Factor(
        "Static analysis", points, 10, True, value,
        "Ruff findings (default rules, run on the indexed version of the file) inside this "
        "code. 0 points for none, 5 for one, 10 for two or more. Existing findings mark code "
        "that deserves a closer look while it is being changed.",
    )


def size_factor(lines_changed: int, whole_file: bool = False) -> Factor:
    if whole_file:
        points, value = 10, "the whole file is removed or renamed"
    else:
        points = 0 if lines_changed <= 0 else 1 if lines_changed <= 3 else 3
        if lines_changed > 10:
            points = 6 if lines_changed <= 30 else 10
        value = f"{lines_changed} line(s) added or removed"
    return Factor(
        "Change size", points, 10, True, value,
        "Lines added plus removed in this code. 1 point for up to 3 lines, 3 for up to 10, "
        "6 for up to 30, 10 above that or when the whole file is removed. Bigger edits carry "
        "more chance of a mistake.",
    )


def heuristic_score(factors: Sequence[Factor]) -> int:
    """0-100, scaled to the factors that could be computed."""
    used = [f for f in factors if f.available]
    maximum = sum(f.max_points for f in used)
    return round(100 * sum(f.points for f in used) / maximum) if maximum else 0


def band_for(score: int) -> str:
    if score >= HIGH_FROM:
        return BAND_HIGH
    if score >= MEDIUM_FROM:
        return BAND_MEDIUM
    return BAND_LOW