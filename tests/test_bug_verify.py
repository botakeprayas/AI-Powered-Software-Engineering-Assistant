import shutil
import tempfile

import pytest
from bug_helpers import (
    BUG_SAMPLE,
    FIX_REPLY,
    HYPOTHESIS_REPLY,
    TEST_REPLY,
    SequenceLLM,
    sample_context,
    sample_traceback,
    validation_config,
)

from core.bugs.investigate import investigate
from core.bugs.verify import (
    DIFFERENT_FAILURE,
    FIX_INEFFECTIVE,
    INCONCLUSIVE,
    NOT_RUN,
    REGRESSION,
    TEST_BROKEN,
    TEST_NOT_REPRODUCING,
    VERIFIED,
    decide,
    mentions_exception,
    verification_to_text,
    verify_investigation,
)
from core.graph.expand import load_graph_context
from core.sandbox.config import SandboxConfig
from core.sandbox.results import PytestSummary, RunResult
from services.llm_service import MockLLM

DRAFT = "tests/test_regression_x.py"
ONE = f"{DRAFT}::test_one"
RELATED = "tests/test_pricing.py::test_average_of_two_items"
ZERO = "E   ZeroDivisionError: division by zero"


def result(outcomes, stdout="", timed_out=False, error=""):
    summary = None if timed_out or error else PytestSummary(passed=1)
    return RunResult(
        ["py"], 1, timed_out, 1.0, stdout=stdout, summary=summary,
        outcomes=outcomes, error=error,
    )


def run_decide(before, after=None, exception="ZeroDivisionError"):
    return decide(before, after, DRAFT, [RELATED], exception)


def test_verified_needs_fail_before_pass_after_and_no_regression():
    before = result({ONE: "FAILED", RELATED: "PASSED"}, ZERO)
    after = result({ONE: "PASSED", RELATED: "PASSED"})
    decision = run_decide(before, after)
    assert decision.verdict == VERIFIED and decision.reproduced
    assert any("1 related test(s) passed before" in reason for reason in decision.reasons)


def test_plain_assert_failures_count_for_assertion_errors():
    before = result({ONE: "FAILED"}, "assert 1 == 2")
    after = result({ONE: "PASSED"})
    assert run_decide(before, after, "AssertionError").verdict == VERIFIED


def test_a_test_that_passes_before_the_fix_does_not_show_the_bug():
    assert run_decide(result({ONE: "PASSED"})).verdict == TEST_NOT_REPRODUCING


def test_a_failure_for_another_reason_is_not_accepted():
    before = result({ONE: "FAILED"}, "E   KeyError: 'x'")
    decision = run_decide(before, result({ONE: "PASSED"}))
    assert decision.verdict == DIFFERENT_FAILURE and not decision.reproduced


def test_an_erroring_draft_is_broken():
    assert run_decide(result({DRAFT: "ERROR"}, ZERO)).verdict == TEST_BROKEN


def test_no_result_for_the_draft_is_inconclusive():
    assert run_decide(result({RELATED: "PASSED"})).verdict == INCONCLUSIVE


def test_no_fix_means_not_run_but_the_bug_is_reproduced():
    decision = run_decide(result({ONE: "FAILED"}, ZERO), None)
    assert decision.verdict == NOT_RUN and decision.reproduced


def test_a_fix_that_leaves_the_draft_failing_is_ineffective():
    before = result({ONE: "FAILED"}, ZERO)
    assert run_decide(before, result({ONE: "FAILED"})).verdict == FIX_INEFFECTIVE


def test_a_fix_that_breaks_a_passing_test_is_a_regression():
    before = result({ONE: "FAILED", RELATED: "PASSED"}, ZERO)
    after = result({ONE: "PASSED", RELATED: "FAILED"})
    decision = run_decide(before, after)
    assert decision.verdict == REGRESSION
    assert RELATED in decision.reasons[-1]


def test_related_tests_that_already_failed_are_not_counted():
    before = result({ONE: "FAILED", RELATED: "FAILED"}, ZERO)
    after = result({ONE: "PASSED", RELATED: "FAILED"})
    decision = run_decide(before, after)
    assert decision.verdict == VERIFIED
    assert any("already failed" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (result({}, timed_out=True), None),
        (result({}, error="could not start"), None),
        (result({ONE: "FAILED"}, ZERO), result({}, timed_out=True)),
        (result({ONE: "FAILED"}, ZERO), result({RELATED: "PASSED"})),
    ],
)
def test_unusable_runs_are_inconclusive(before, after):
    assert run_decide(before, after).verdict == INCONCLUSIVE


def test_mentions_exception():
    assert mentions_exception("ZeroDivisionError", ZERO) is True
    assert mentions_exception("builtins.KeyError", "E   KeyError: 'x'") is True
    assert mentions_exception("KeyError", ZERO) is False
    assert mentions_exception("", ZERO) is None
    assert mentions_exception("AssertionError", "assert 1 == 2") is True


# ---- real sandbox runs (real subprocesses) ---------------------------------------------


def lines(*parts):
    return "\n".join(parts) + "\n"


def fix_reply(replacement):
    return lines(
        "RATIONALE: change the return.",
        "FILE: cart/pricing.py",
        "<<<<<<< SEARCH",
        "    return total / len(items)",
        "=======",
        replacement,
        ">>>>>>> REPLACE",
    )


def test_reply(name, *code):
    body = "\n".join(code)
    return f"FILE: tests/test_regression_{name}.py\n```python\n{body}\n```\n"


test_reply.__test__ = False  # a helper, not a test: stop pytest from collecting it


INEFFECTIVE_FIX = fix_reply("    return total / (len(items) or 0)")
BREAKING_FIX = fix_reply("    return 0.0")
PASSING_TEST = test_reply(
    "ok",
    "from cart.pricing import average_price", "", "",
    "def test_single_item():",
    "    assert average_price([('a', 2.0)]) == 2.0",
)
BROKEN_TEST = test_reply(
    "broken",
    "from cart.pricing import not_there", "", "",
    "def test_x():",
    "    assert not_there([]) == 0",
)
SLOW_TEST = test_reply(
    "slow",
    "import time", "", "",
    "def test_slow():",
    "    time.sleep(60)",
)


@pytest.fixture(scope="module")
def context():
    return sample_context()


def investigate_with(context, replies):
    llm = SequenceLLM(replies)
    return investigate(
        sample_traceback(), context, llm, validation_config=validation_config()
    )


def snapshot_files():
    return {str(p): p.read_bytes() for p in BUG_SAMPLE.rglob("*") if p.is_file()}


def test_a_good_fix_is_verified_from_real_pytest_output(context):
    before_files = snapshot_files()
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    report = verify_investigation(inv, BUG_SAMPLE, SandboxConfig(timeout_seconds=60))
    assert report.verdict == VERIFIED, verification_to_text(report)
    draft_test = "tests/test_regression_average_price.py::test_average_of_empty_cart_is_zero"
    assert report.before.outcomes[draft_test] == "FAILED"
    assert report.before.outcomes[RELATED] == "PASSED"
    assert "ZeroDivisionError" in report.before.stdout
    assert report.after.outcomes[draft_test] == "PASSED"
    assert report.after.outcomes[RELATED] == "PASSED"
    assert snapshot_files() == before_files  # the project folder was never touched
    assert "VERIFICATION: VERIFIED" in verification_to_text(report)


def test_a_fix_that_does_not_fix_the_bug_is_ineffective(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, INEFFECTIVE_FIX, TEST_REPLY])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == FIX_INEFFECTIVE, verification_to_text(report)
    assert report.reproduced


def test_a_fix_that_breaks_an_existing_test_is_a_regression(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, BREAKING_FIX, TEST_REPLY])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == REGRESSION, verification_to_text(report)
    assert RELATED in report.reasons[-1]


def test_a_test_that_already_passes_does_not_reproduce_the_bug(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, PASSING_TEST])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == TEST_NOT_REPRODUCING, verification_to_text(report)
    assert report.after is None  # the fix was not even tried


def test_a_test_with_an_import_error_is_broken(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, BROKEN_TEST])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == TEST_BROKEN, verification_to_text(report)


def test_a_runaway_test_times_out_and_is_inconclusive(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, SLOW_TEST])
    report = verify_investigation(inv, BUG_SAMPLE, SandboxConfig(timeout_seconds=4))
    assert report.verdict == INCONCLUSIVE
    assert report.before.timed_out
    assert any("timed out" in reason for reason in report.reasons)


def test_with_no_fix_the_bug_is_still_reproduced(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, "NO_FIX: need more evidence", TEST_REPLY])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == NOT_RUN and report.reproduced
    assert report.after is None and report.before is not None


def test_the_mock_provider_runs_nothing(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY])
    inv.is_mock = True
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == NOT_RUN and report.before is None


def test_generated_test_code_cannot_read_host_files(context, tmp_path):
    secret = tmp_path / "host_secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    spy = test_reply(
        "hostread",
        f"HOST = {str(secret)!r}", "", "",
        "def test_read():",
        "    open(HOST, encoding='utf-8').read()",
    )
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, spy])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.before.outcomes["tests/test_regression_hostread.py::test_read"] == "FAILED"
    assert "blocked" in report.before.stdout
    assert "TOP SECRET" not in report.before.stdout + report.before.stderr
    assert report.verdict == DIFFERENT_FAILURE


def test_a_fix_that_no_longer_matches_the_files_is_inconclusive(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(BUG_SAMPLE, repo)
    inv = investigate_with(load_graph_context(repo), [HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    pricing = repo / "cart" / "pricing.py"
    text = pricing.read_text(encoding="utf-8")
    changed = text.replace("sum(price for _, price in items)", "sum(p for _, p in items)")
    assert changed != text
    pricing.write_text(changed, encoding="utf-8")
    report = verify_investigation(inv, repo)
    assert report.verdict == INCONCLUSIVE, verification_to_text(report)
    assert any("no longer applies" in reason for reason in report.reasons)


def test_sandbox_folders_are_deleted_afterwards(context, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, TEST_REPLY])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == VERIFIED
    assert list(tmp_path.glob("ai_se_sandbox_*")) == []


def test_verifying_a_mock_investigation_never_crashes(context):
    llm = MockLLM()
    inv = investigate(sample_traceback(), context, llm, validation_config=validation_config())
    assert verify_investigation(inv, BUG_SAMPLE).verdict == NOT_RUN

NONE_FIX = fix_reply(
    "    if len(items) == 0:\n        return None\n    return total / len(items)"
)
SUMMARY_TEST = test_reply(
    "empty_summary",
    "from cart.checkout import summary", "", "",
    "def test_summary_of_an_empty_cart_does_not_crash():",
    "    summary([])",
)


def test_a_fix_that_only_moves_the_crash_to_the_caller_is_ineffective(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, NONE_FIX, SUMMARY_TEST])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == FIX_INEFFECTIVE, verification_to_text(report)
    assert "TypeError" in report.after.stdout


def test_the_same_test_verifies_a_fix_that_really_works(context):
    inv = investigate_with(context, [HYPOTHESIS_REPLY, FIX_REPLY, SUMMARY_TEST])
    report = verify_investigation(inv, BUG_SAMPLE)
    assert report.verdict == VERIFIED, verification_to_text(report)