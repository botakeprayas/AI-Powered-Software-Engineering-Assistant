import pytest
from impact_e2e_helpers import (
    FakeSignals,
    analyze_diff,
    edit_diff,
    sample_text,
    unused_variable,
)
from impact_helpers import make_delete_diff, make_diff

from core.impact.analyze import (
    ImpactConfig,
    compare_files,
    compare_tests,
    load_impact_config,
)
from core.impact.rank import BAND_LOW, BAND_MEDIUM, HEURISTIC_DISCLAIMER

RATES = "pricing/rates.py"
TAX_BODY = "    return round(amount * (1 + TAX_RATE), 3)"
KNOWN_TESTS = [
    "tests/test_rates.py::test_apply_tax",
    "tests/test_invoice.py::test_invoice_total",
    "tests/test_invoice.py::test_tax_only",
    "tests/test_checkout.py::test_checkout",
]
ENV_NAMES = (
    "IMPACT_MAX_DEPTH", "IMPACT_MAX_DEPENDENTS", "IMPACT_CHURN_DAYS", "IMPACT_USE_GIT",
    "IMPACT_USE_RUFF",
)


def signals():
    return FakeSignals(churn={RATES: 7}, findings={RATES: [unused_variable()]})


def factor(impact, name):
    return next(f for f in impact.factors if f.name == name)


@pytest.fixture(scope="module")
def tax_report():
    return analyze_diff(edit_diff(RATES, 8, TAX_BODY), signals())


def test_editing_apply_tax_finds_the_affected_files_known_by_hand(tax_report):
    assert tax_report.changed_files == [RATES]
    assert [u.unit.qualname for u in tax_report.units] == ["apply_tax"]
    assert tax_report.known_file_paths == ["app.py", "pricing/invoice.py", "shop/checkout.py"]
    assert tax_report.inferred_file_paths == []
    everything = set(tax_report.known_file_paths)
    assert not everything & {"tools/formatting.py", "shop/receipt.py", "shop/display.py"}


def test_recommended_tests_are_the_ones_that_reach_the_change_closest_first(tax_report):
    assert [t.id for t in tax_report.tests] == [
        "tests/test_rates.py::test_apply_tax",
        "tests/test_invoice.py::test_invoice_total",
        "tests/test_invoice.py::test_tax_only",
        "tests/test_checkout.py::test_checkout",
    ]
    assert {t.status for t in tax_report.tests} == {"known"}
    assert "tests/test_rates.py::test_apply_discount" not in tax_report.known_test_ids
    assert tax_report.test_command.startswith("pytest tests/test_rates.py::test_apply_tax ")


def test_every_factor_is_shown_with_its_points_and_the_score_is_a_heuristic_rank(tax_report):
    (impact,) = tax_report.units
    points = {f.name: f.points for f in impact.factors}
    assert points == {
        "Dependents": 30, "Graph position": 20, "Test reach": 0, "Code churn": 10,
        "Static analysis": 0, "Change size": 1,
    }
    assert (impact.score, impact.band, impact.rank) == (61, BAND_MEDIUM, 1)
    assert tax_report.disclaimer == HEURISTIC_DISCLAIMER
    assert "not probabilities" in tax_report.disclaimer
    assert all(f.explanation for f in impact.factors)
    assert not any("probab" in f.explanation.lower() for f in impact.factors)


def test_affected_files_carry_reasons_with_source_lines(tax_report):
    by_path = {f.path: f for f in tax_report.affected_known}
    assert by_path["pricing/invoice.py"].min_depth == 1
    assert by_path["pricing/invoice.py"].labels == ("invoice_total", "tax_only")
    assert by_path["shop/checkout.py"].reasons[0].startswith(
        "checkout calls invoice_total at shop/checkout.py:7 (confirmed)"
    )
    assert by_path["app.py"].min_depth == 3 and not by_path["app.py"].also_changed


def test_an_edit_inside_apply_discount_does_not_touch_tax_only_and_counts_ruff_findings():
    report = analyze_diff(edit_diff(RATES, 14, "    return amount"), signals())
    (impact,) = report.units
    assert impact.unit.qualname == "apply_discount"
    assert factor(impact, "Static analysis").points == 5
    assert "F841" in factor(impact, "Static analysis").value
    labels = {d.label for d in impact.traversal.dependents}
    assert "tax_only" not in labels and "invoice_total" in labels
    assert report.known_file_paths == ["app.py", "pricing/invoice.py", "shop/checkout.py"]


def test_inferred_impacts_are_kept_apart_from_known_ones():
    report = analyze_diff(edit_diff("shop/display.py", 9, '        print(f"{value}")'), signals())
    (impact,) = report.units
    assert impact.unit.qualname == "ConsoleDisplay.show"
    assert report.known_file_paths == [] and report.inferred_file_paths == ["shop/receipt.py"]
    assert report.affected_inferred[0].reasons[0].startswith(
        "print_receipt calls `printer.show` at shop/receipt.py:6 (UNCERTAIN"
    )
    assert report.tests == []  # no test reaches it, and none is named after display.py
    assert factor(impact, "Test reach").points == 20
    assert factor(impact, "Dependents").value == "0 known, 1 inferred"
    assert (impact.score, impact.band) == (24, BAND_LOW)
    wanted = "inferred ones come from calls on objects of unknown type"
    assert any(wanted in n for n in impact.notes)
    assert any("No test reaches" in n for n in impact.notes)


def test_a_changed_constant_reaches_only_the_importers_and_says_what_is_not_tracked():
    report = analyze_diff(edit_diff(RATES, 3, "TAX_RATE = 0.1"), signals())
    (impact,) = report.units
    assert impact.unit.is_module_level
    assert report.known_file_paths == ["pricing/invoice.py"]
    assert report.known_test_ids == ["tests/test_rates.py"]
    assert any("not tracked" in note for note in impact.notes)
    by_hand = ["pricing/invoice.py", "shop/checkout.py"]  # apply_tax uses the constant
    comparison = compare_files(report, by_hand)
    assert comparison.found_known == ("pricing/invoice.py",)
    assert comparison.missed == ("shop/checkout.py",)
    assert comparison.recall_known == 0.5
    assert "1 of 2 expected items were found as known" in comparison.summary()


def test_an_unrelated_change_affects_nothing():
    report = analyze_diff(edit_diff("tools/formatting.py", 6, "    return text.lower()"), signals())
    assert report.affected_known == [] and report.affected_inferred == []
    assert report.tests == []
    (impact,) = report.units
    assert factor(impact, "Dependents").points == 0
    assert factor(impact, "Graph position").points == 0


def test_a_deleted_file_changes_every_symbol_and_every_importer():
    diff = make_delete_diff("shop/checkout.py", sample_text("shop/checkout.py"))
    report = analyze_diff(diff, signals())
    assert {u.unit.qualname for u in report.units} == {"checkout", "module shop/checkout.py"}
    assert report.known_file_paths == ["app.py"]
    assert report.known_test_ids == ["tests/test_checkout.py::test_checkout"]
    assert all(factor(u, "Change size").points == 10 for u in report.units)
    assert all(any("fails at import time" in n for n in u.notes) for u in report.units)


def test_the_definition_line_edit_is_flagged_as_a_possible_api_change():
    sig = analyze_diff(edit_diff(RATES, 6, "def apply_tax(amount, rate=None):"), signals())
    assert sig.units[0].signature_changed and sig.units[0].public
    assert any("callers may need to change" in n for n in sig.units[0].notes)
    body = analyze_diff(edit_diff(RATES, 8, TAX_BODY), signals())
    assert not body.units[0].signature_changed


def test_missing_git_and_ruff_leave_those_factors_out_and_scale_the_score():
    report = analyze_diff(
        edit_diff(RATES, 8, TAX_BODY), FakeSignals(git=False, ruff=False)
    )
    (impact,) = report.units
    assert not factor(impact, "Code churn").available
    assert not factor(impact, "Static analysis").available
    assert impact.score == round(100 * (30 + 20 + 0 + 1) / 80)
    assert any("churn is not scored" in n for n in report.notes)
    assert any("findings are not scored" in n for n in report.notes)


def test_a_diff_that_maps_to_nothing_gives_an_honest_empty_report():
    report = analyze_diff("diff --git a/README.md b/README.md\n--- a/README.md\n"
                          "+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n")
    assert report.units == [] and report.tests == []
    assert report.skipped and report.skipped[0][0] == "README.md"
    assert any("No changed Python code could be mapped" in n for n in report.notes)


def test_two_changed_units_are_ranked_highest_first():
    old = sample_text(RATES)
    lines = old.split("\n")
    lines[7] = TAX_BODY
    lines[13] = "    return amount"
    report = analyze_diff(make_diff(RATES, old, "\n".join(lines)), signals())
    assert {u.unit.qualname for u in report.units} == {"apply_tax", "apply_discount"}
    assert [u.rank for u in report.units] == [1, 2]
    assert report.units[0].score >= report.units[1].score

def test_predictions_can_be_compared_with_a_hand_checked_list(tax_report):
    files = compare_files(tax_report, ["pricing/invoice.py", "shop/checkout.py", "app.py"])
    assert files.found_known == ("app.py", "pricing/invoice.py", "shop/checkout.py")
    assert files.missed == () and files.extra_known == ()
    assert (files.recall_known, files.precision_known) == (1.0, 1.0)
    tests = compare_tests(tax_report, KNOWN_TESTS[:3] + ["tests/test_other.py::test_x"])
    assert tests.missed == ("tests/test_other.py::test_x",)
    assert tests.extra_known == ("tests/test_checkout.py::test_checkout",)
    assert compare_files(tax_report, []).recall_known is None


def test_config_defaults_and_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert load_impact_config(env_file=None) == ImpactConfig()
    monkeypatch.setenv("IMPACT_MAX_DEPTH", "2")
    monkeypatch.setenv("IMPACT_USE_GIT", "off")
    config = load_impact_config(env_file=None)
    assert config.traversal.max_depth == 2 and config.use_git is False


@pytest.mark.parametrize(
    ("name", "value"),
    [("IMPACT_MAX_DEPTH", "9"), ("IMPACT_CHURN_DAYS", "soon"), ("IMPACT_USE_RUFF", "maybe"),
     ("IMPACT_CHURN_DAYS", "0")],
)
def test_config_rejects_bad_values(monkeypatch, name, value):
    for env_name in ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="IMPACT_"):
        load_impact_config(env_file=None)
        
def test_a_churn_window_longer_than_ten_years_is_rejected(monkeypatch):
    for env_name in ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("IMPACT_CHURN_DAYS", "5000")
    with pytest.raises(ValueError, match="IMPACT_CHURN_DAYS"):
        load_impact_config(env_file=None)