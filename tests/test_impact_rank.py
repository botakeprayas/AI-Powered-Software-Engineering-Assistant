import pytest

from core.impact.rank import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    HEURISTIC_DISCLAIMER,
    band_for,
    churn_factor,
    dependents_factor,
    heuristic_score,
    lint_factor,
    position_factor,
    size_factor,
    test_factor,
)


@pytest.mark.parametrize(
    ("known", "inferred", "points"),
    [(0, 0, 0), (1, 0, 10), (2, 1, 23), (3, 0, 30), (5, 5, 30), (0, 2, 6)],
)
def test_dependents_points(known, inferred, points):
    assert dependents_factor(known, inferred).points == points


@pytest.mark.parametrize(
    ("depth", "module", "points"),
    [(0, False, 0), (0, True, 0), (1, False, 5), (2, False, 10), (3, False, 15),
     (1, True, 10), (4, True, 20), (5, True, 20)],
)
def test_position_points(depth, module, points):
    assert position_factor(depth, module, False, 4).points == points


def test_position_value_mentions_a_stopped_trace():
    assert "trace stopped at depth 4" in position_factor(4, False, True, 4).value


@pytest.mark.parametrize(
    ("tests", "points"), [(0, 20), (1, 10), (2, 5), (3, 0), (9, 0)]
)
def test_test_reach_points(tests, points):
    assert test_factor(tests).points == points


@pytest.mark.parametrize(
    ("commits", "points"), [(0, 0), (1, 3), (2, 3), (3, 6), (5, 6), (6, 10), (40, 10)]
)
def test_churn_points(commits, points):
    factor = churn_factor(commits, 180)
    assert factor.available and factor.points == points


@pytest.mark.parametrize(
    ("codes", "points"), [([], 0), (["F841"], 5), (["F841", "F401"], 10), (["E"] * 6, 10)]
)
def test_lint_points(codes, points):
    assert lint_factor(codes).points == points


@pytest.mark.parametrize(
    ("lines", "whole", "points"),
    [(0, False, 0), (2, False, 1), (3, False, 1), (4, False, 3), (10, False, 3),
     (11, False, 6), (30, False, 6), (31, False, 10), (1, True, 10)],
)
def test_size_points(lines, whole, points):
    assert size_factor(lines, whole).points == points


def test_unavailable_factors_are_flagged_and_left_out_of_the_score():
    assert not churn_factor(None, 180).available
    assert not lint_factor(None).available
    full = [dependents_factor(3, 0), position_factor(1, False, False, 4), test_factor(0),
            churn_factor(0, 180), lint_factor([]), size_factor(1)]
    assert heuristic_score(full) == round(100 * (30 + 5 + 20 + 0 + 0 + 1) / 100)
    reduced = [dependents_factor(3, 0), position_factor(1, False, False, 4), test_factor(0),
               churn_factor(None, 180), lint_factor(None), size_factor(1)]
    assert heuristic_score(reduced) == round(100 * (30 + 5 + 20 + 1) / 80)


def test_bands_and_disclaimer():
    assert [band_for(s) for s in (0, 33, 34, 66, 67, 100)] == [
        BAND_LOW, BAND_LOW, BAND_MEDIUM, BAND_MEDIUM, BAND_HIGH, BAND_HIGH,
    ]
    assert "not probabilities" in HEURISTIC_DISCLAIMER
    assert "heuristic" in HEURISTIC_DISCLAIMER


def test_every_factor_explains_itself():
    factors = [dependents_factor(1, 1), position_factor(2, True, False, 4), test_factor(1),
               churn_factor(4, 180), lint_factor(["F841"]), size_factor(5)]
    assert [f.name for f in factors] == [
        "Dependents", "Graph position", "Test reach", "Code churn", "Static analysis",
        "Change size",
    ]
    assert all(f.value and len(f.explanation) > 40 and f.max_points > 0 for f in factors)