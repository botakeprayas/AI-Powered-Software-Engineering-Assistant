from impact_e2e_helpers import FakeSignals, analyze_diff, edit_diff, unused_variable

from core.impact.analyze import compare_files, compare_tests
from core.impact.render import (
    dependent_rows,
    factor_rows,
    file_rows,
    recommended_rows,
    report_to_text,
    skipped_rows,
    unit_rows,
)

RATES = "pricing/rates.py"
TAX_BODY = "    return round(amount * (1 + TAX_RATE), 3)"
TESTS = [
    "tests/test_rates.py::test_apply_tax",
    "tests/test_invoice.py::test_invoice_total",
    "tests/test_invoice.py::test_tax_only",
    "tests/test_checkout.py::test_checkout",
]


def signals():
    return FakeSignals(churn={RATES: 7}, findings={RATES: [unused_variable()]})


def tax_report():
    return analyze_diff(edit_diff(RATES, 8, TAX_BODY), signals())


def display_report():
    diff = edit_diff("shop/display.py", 9, '        print(f"{value}")')
    return analyze_diff(diff, signals())


def test_the_text_report_shows_every_factor_with_its_points_and_a_heuristic_label():
    text = report_to_text(tax_report())
    assert "heuristic rank 61 of 100 (medium)" in text
    assert "Dependents: 30/30 points | 5 known, 0 inferred" in text
    assert "Graph position: 20/20 points" in text
    assert "Test reach: 0/20 points | 4 test(s) reach it" in text
    assert "Code churn: 10/10 points | 7 commit(s) in 180 days" in text
    assert "Change size: 1/10 points | 2 line(s) added or removed" in text
    assert "not probabilities" in text
    assert "what changed: lines 8 changed" in text
    assert "Commits that touched this file" in text  # the plain explanation of a factor


def test_the_text_report_separates_known_dependencies_from_inferred_impacts():
    text = report_to_text(tax_report())
    known = text.split("KNOWN DEPENDENCIES")[1].split("INFERRED IMPACTS")[0]
    assert "app.py (depth 3)" in known and "pricing/invoice.py (depth 1)" in known
    assert "why: checkout calls invoice_total at shop/checkout.py:7 (confirmed)" in known
    inferred = text.split("INFERRED IMPACTS")[1].split("RECOMMENDED TESTS")[0]
    assert "(none found)" in inferred
    assert "run: pytest tests/test_rates.py::test_apply_tax" in text


def test_an_inferred_impact_is_labeled_not_confirmed_and_names_the_uncertain_call():
    text = report_to_text(display_report())
    inferred = text.split("INFERRED IMPACTS")[1].split("RECOMMENDED TESTS")[0]
    assert "shop/receipt.py (depth 1)" in inferred
    assert "UNCERTAIN: ConsoleDisplay.show is one possible target, not confirmed" in inferred
    known = text.split("KNOWN DEPENDENCIES")[1].split("INFERRED IMPACTS")[0]
    assert "(none found)" in known
    assert "note: No test reaches this code through confirmed edges." in text
    assert "(none found: no test reaches the changed code" in text


def test_unavailable_factors_are_reported_as_not_computed():
    report = analyze_diff(
        edit_diff(RATES, 8, TAX_BODY), FakeSignals(git=False, ruff=False)
    )
    text = report_to_text(report)
    assert "Code churn: not computed (left out of the score) | not available" in text
    assert "Static analysis: not computed (left out of the score)" in text
    assert "Git history was not available, so churn is not scored." in text


def test_a_diff_that_maps_to_nothing_prints_an_honest_report():
    readme = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    text = report_to_text(analyze_diff(readme))
    assert "No changed Python code could be mapped" in text
    assert "NOT ANALYZED" in text and "README.md: not a Python file" in text
    assert "WHAT THIS ANALYSIS CANNOT SEE" in text


def test_comparisons_with_a_hand_checked_list_are_printed():
    report = tax_report()
    comparisons = {
        "affected files": compare_files(report, ["pricing/invoice.py", "shop/checkout.py"]),
        "tests": compare_tests(report, [*TESTS[:3], "tests/test_other.py::test_x"]),
    }
    text = report_to_text(report, comparisons)
    assert "COMPARISON WITH YOUR HAND-CHECKED LISTS" in text
    assert "known but not on your list: app.py" in text
    assert "missed: tests/test_other.py::test_x" in text
    assert "recall 100%" in text and "recall 75%" in text


def test_the_table_rows_carry_the_same_facts_as_the_text():
    report = tax_report()
    (row,) = unit_rows(report)
    assert row["rank"] == 1 and row["changed code"] == "apply_tax"
    assert row["where"] == "pricing/rates.py:6-8" and row["band"] == "medium"
    assert (row["known dependents"], row["inferred dependents"]) == (5, 0)
    assert row["tests reaching it"] == 4 and row["heuristic rank (0-100)"] == 61
    factors = {r["factor"]: r["points"] for r in factor_rows(report.units[0])}
    assert factors["Dependents"] == "30/30" and factors["Test reach"] == "0/20"
    files = file_rows(report.affected_known)
    assert [r["file"] for r in files] == ["app.py", "pricing/invoice.py", "shop/checkout.py"]
    assert all(r["why"] for r in files)
    assert [r["test"] for r in recommended_rows(report.tests)] == TESTS
    deps = dependent_rows(report.units[0].known)
    assert {r["code"] for r in deps} == {"invoice_total", "tax_only", "checkout", "main", "app.py"}
    assert skipped_rows(report) == []


def test_the_report_never_calls_a_score_a_probability_or_a_percentage():
    text = report_to_text(tax_report())
    assert "probability" not in text.replace("not probabilities", "").lower()
    assert "% likely" not in text and "chance of failure" not in text.lower()