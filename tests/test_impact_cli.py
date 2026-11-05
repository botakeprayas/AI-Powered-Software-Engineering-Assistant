import importlib.util
import io
from pathlib import Path

import pytest
from impact_e2e_helpers import edit_diff

ROOT = Path(__file__).resolve().parents[1]
REPO = str(ROOT / "data" / "impact_sample")
RATES = "pricing/rates.py"
TAX_BODY = "    return round(amount * (1 + TAX_RATE), 3)"
FAST = ["--no-git", "--no-ruff"]
TESTS = (
    "tests/test_rates.py::test_apply_tax,tests/test_invoice.py::test_invoice_total,"
    "tests/test_invoice.py::test_tax_only,tests/test_checkout.py::test_checkout"
)


def load_cli():
    spec = importlib.util.spec_from_file_location("impact_cli", ROOT / "scripts" / "impact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_cli()


@pytest.fixture
def diff_file(tmp_path):
    path = tmp_path / "apply_tax.diff"
    path.write_text(edit_diff(RATES, 8, TAX_BODY), encoding="utf-8")
    return str(path)


def test_the_cli_prints_a_report_and_compares_with_hand_checked_lists(diff_file, capsys):
    code = CLI.main(
        ["--repo", REPO, "--diff-file", diff_file, *FAST,
         "--expect-files", "pricing/invoice.py,shop/checkout.py,app.py",
         "--expect-tests", TESTS]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "CHANGE-IMPACT REPORT" in out and "not probabilities" in out
    assert "1. apply_tax [function] pricing/rates.py:6-8" in out
    assert "app.py (depth 3)" in out and "pricing/invoice.py (depth 1)" in out
    assert "affected files: 3 of 3 expected items were found as known (recall 100%)" in out
    assert "tests: 4 of 4 expected items were found as known (recall 100%)" in out
    assert "not computed (left out of the score)" in out


def test_the_cli_can_read_the_diff_from_standard_input(diff_file, capsys, monkeypatch):
    text = Path(diff_file).read_text(encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    assert CLI.main(["--repo", REPO, "--diff-file", "-", *FAST]) == 0
    assert "apply_tax" in capsys.readouterr().out


def test_a_hand_checked_miss_is_reported_not_hidden(tmp_path, capsys):
    path = tmp_path / "constant.diff"
    path.write_text(edit_diff(RATES, 3, "TAX_RATE = 0.1"), encoding="utf-8")
    code = CLI.main(
        ["--repo", REPO, "--diff-file", str(path), *FAST,
         "--expect-files", "pricing/invoice.py,shop/checkout.py"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "1 of 2 expected items were found as known (recall 50%)" in out
    assert "missed: shop/checkout.py" in out
    assert "not tracked" in out


def test_a_diff_with_nothing_to_map_exits_with_3(capsys):
    assert CLI.main(["--repo", REPO, "--diff", "", *FAST]) == 3
    assert "No changed Python code could be mapped" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--repo", REPO, "--diff-file", "does_not_exist.diff"],
        ["--repo", "no_such_folder_anywhere", "--diff", "x"],
        ["--repo", REPO, "--diff", "x", "--depth", "0"],
    ],
)
def test_bad_input_gives_a_clear_error_and_exit_code_1(args, capsys):
    assert CLI.main([*args, *FAST]) == 1
    assert "ERROR:" in capsys.readouterr().err


def test_a_leading_byte_order_mark_in_the_diff_file_is_accepted(tmp_path, capsys):
    path = tmp_path / "bom.diff"
    path.write_text(edit_diff(RATES, 8, TAX_BODY), encoding="utf-8-sig")
    assert CLI.main(["--repo", REPO, "--diff-file", str(path), *FAST]) == 0
    assert "1. apply_tax" in capsys.readouterr().out