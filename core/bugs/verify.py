"""Verification (Phase 6, part 3): run the draft test before and after the proposed fix.

Both runs happen in disposable sandboxes (temporary copies of the repository). Pass/fail is
read from real pytest output. "Verified" has a narrow meaning: the draft test failed on the
current code, passes with the fix, and related tests that passed before still pass. It does
not prove the fix is correct in general, and the test itself was written by a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.bugs.investigate import Investigation
from core.bugs.proposal import Edit
from core.sandbox.config import SandboxConfig
from core.sandbox.results import RunResult
from core.sandbox.runner import Sandbox, SandboxError

VERIFIED = "verified"
FIX_INEFFECTIVE = "fix_ineffective"
REGRESSION = "regression"
TEST_NOT_REPRODUCING = "test_not_reproducing"
DIFFERENT_FAILURE = "different_failure"
TEST_BROKEN = "test_broken"
INCONCLUSIVE = "inconclusive"
NOT_RUN = "not_run"

VERDICT_TEXT = {
    VERIFIED: "VERIFIED by these tests: the draft test failed before the fix and passes after it.",
    FIX_INEFFECTIVE: "NOT VERIFIED: the draft test still fails (or errors) after the fix.",
    REGRESSION: "NOT VERIFIED: the fix makes a related test fail that passed before.",
    TEST_NOT_REPRODUCING: "NOT VERIFIED: the draft test passes on the current code, "
    "so it does not show the bug.",
    DIFFERENT_FAILURE: "NOT VERIFIED: the draft test fails, but possibly for a different "
    "reason than the reported bug.",
    TEST_BROKEN: "NOT VERIFIED: the draft test errors before the fix, so it cannot show the bug.",
    INCONCLUSIVE: "NOT VERIFIED: the runs gave no trustworthy result.",
    NOT_RUN: "NOT VERIFIED: nothing could be run, or there was no fix to verify.",
}
LIMITS_TEXT = (
    "Verified means only this: the draft test failed on the current code, passes with the "
    "fix, and related tests that passed before still pass. The test was written by a model "
    "and may not encode the right behavior, and passing tests do not prove a fix is correct. "
    "Read the diff and the test."
)
SANDBOX_TEXT = (
    "Tests ran in a temporary copy of the repository with a minimal environment, a timeout, "
    "and a best-effort guard against file access outside the copy, network access and "
    "starting programs. This is not a security boundary: use Docker or a virtual machine "
    "for untrusted code. CPU and memory are not limited."
)


@dataclass(frozen=True)
class Decision:
    verdict: str
    reasons: tuple[str, ...]
    reproduced: bool = False  # the draft test failed on the current code for the reported reason


@dataclass
class VerificationReport:
    verdict: str = NOT_RUN
    reasons: list[str] = field(default_factory=list)
    reproduced: bool = False
    draft_path: str = ""
    test_ids: list[str] = field(default_factory=list)
    before: RunResult | None = None  # real pytest result on the current code
    after: RunResult | None = None  # real pytest result with the proposed fix
    notes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.verdict == VERIFIED

    @property
    def headline(self) -> str:
        return VERDICT_TEXT.get(self.verdict, self.verdict)


def mentions_exception(exception_type: str, text: str) -> bool | None:
    """Does the output mention the reported exception? None when the type is unknown."""
    name = exception_type.strip().rsplit(".", 1)[-1]
    if not name:
        return None
    if name in text:
        return True
    return name == "AssertionError" and "assert " in text  # plain asserts print 'assert x == y'


def draft_outcomes(result: RunResult, draft_path: str) -> dict[str, str]:
    """pytest outcomes that belong to the draft test file."""
    prefix = draft_path + "::"
    return {
        test: outcome for test, outcome in result.outcomes.items()
        if test == draft_path or test.startswith(prefix)
    }


def _unusable(result: RunResult, when: str) -> str:
    if result.error:
        return f"The sandbox could not run pytest {when}: {result.error}"
    if result.timed_out:
        return f"pytest timed out {when} after {result.duration:.0f}s, so there is no result."
    if result.summary is None and not result.outcomes:
        return f"pytest printed no summary {when}, so there is no trustworthy result."
    return ""


def decide(
    before: RunResult,
    after: RunResult | None,
    draft_path: str,
    related_ids: list[str],
    exception_type: str = "",
) -> Decision:
    """Turn two real pytest results into a verdict. Pure: it only reads the results."""
    problem = _unusable(before, "before the fix")
    if problem:
        return Decision(INCONCLUSIVE, (problem,))
    drafts = draft_outcomes(before, draft_path)
    if "ERROR" in drafts.values():
        reason = (
            "The draft test errors before the fix (for example an import error), "
            "so it cannot show the bug."
        )
        return Decision(TEST_BROKEN, (reason,))
    failed = sorted(test for test, outcome in drafts.items() if outcome == "FAILED")
    if not failed:
        if "PASSED" in drafts.values():
            reason = "The draft test passes on the current code, so it does not show the bug."
            return Decision(TEST_NOT_REPRODUCING, (reason,))
        return Decision(INCONCLUSIVE, ("pytest reported no result for the draft test.",))
    mention = mentions_exception(exception_type, f"{before.stdout}\n{before.stderr}")
    if mention is False:
        reason = (
            f"The draft test fails before the fix, but its output never mentions "
            f"{exception_type}, so it may fail for a different reason than the reported bug."
        )
        return Decision(DIFFERENT_FAILURE, (reason,))
    reasons = [f"The draft test fails before the fix: {', '.join(failed)}."]
    if mention:
        reasons.append(f"The failure output mentions {exception_type}, as reported.")
    if after is None:
        return Decision(NOT_RUN, (*reasons, "There is no fix to verify."), True)

    problem = _unusable(after, "after the fix")
    if problem:
        return Decision(INCONCLUSIVE, (*reasons, problem), True)
    after_drafts = draft_outcomes(after, draft_path)
    if not after_drafts:
        reason = "pytest reported no result for the draft test after the fix."
        return Decision(INCONCLUSIVE, (*reasons, reason), True)
    stuck = sorted(test for test, outcome in after_drafts.items() if outcome != "PASSED")
    if stuck:
        reason = "The draft test still fails (or errors) after the fix: " + ", ".join(stuck) + "."
        return Decision(FIX_INEFFECTIVE, (*reasons, reason), True)
    missing = [test for test in failed if test not in after_drafts]
    if missing:
        reason = "pytest reported no result after the fix for: " + ", ".join(missing) + "."
        return Decision(INCONCLUSIVE, (*reasons, reason), True)
    regressions = [
        test for test in related_ids
        if before.outcomes.get(test) == "PASSED" and after.outcomes.get(test) != "PASSED"
    ]
    if regressions:
        reason = "Related test(s) passed before the fix but not after it: " + ", ".join(
            regressions
        ) + "."
        return Decision(REGRESSION, (*reasons, reason), True)
    kept = [test for test in related_ids if before.outcomes.get(test) == "PASSED"]
    already = [test for test in related_ids if before.outcomes.get(test) in ("FAILED", "ERROR")]
    reasons.append("The draft test passes after the fix.")
    reasons.append(f"{len(kept)} related test(s) passed before and still pass after.")
    if already:
        reasons.append(
            "Related test(s) that already failed before the fix were not counted: "
            + ", ".join(already) + "."
        )
    return Decision(VERIFIED, tuple(reasons), True)


def apply_edits(box: Sandbox, edits: list[Edit]) -> str:
    """Re-apply the fix's edits to the sandbox copy. Returns '' or the reason it failed."""
    texts: dict[str, str] = {}
    for edit in edits:
        if edit.file not in texts:
            try:
                texts[edit.file] = box.read_text(edit.file)
            except (OSError, UnicodeDecodeError, SandboxError):
                return f"{edit.file} could not be read in the sandbox copy."
        count = texts[edit.file].count(edit.search)
        if count != 1:
            return (
                f"The fix no longer applies: its SEARCH text for {edit.file} matches "
                f"{count} place(s) in the current file."
            )
        texts[edit.file] = texts[edit.file].replace(edit.search, edit.replace, 1)
    for name, text in texts.items():
        box.write_file(name, text)
    return ""


def _finish(report: VerificationReport, decision: Decision) -> VerificationReport:
    report.verdict = decision.verdict
    report.reasons = list(decision.reasons)
    report.reproduced = decision.reproduced
    return report


def _copy_note(box: Sandbox) -> str:
    copied = box.copy_report
    return (
        f"Sandbox copy: {copied.files} file(s) copied; {len(copied.skipped)} not copied "
        "(secrets, environments, caches and generated files are never copied)."
    )


def verify_investigation(
    inv: Investigation, repo_root: Path, config: SandboxConfig | None = None
) -> VerificationReport:
    """Run the draft test on the current code, then with the proposed fix. Never raises."""
    config = config or SandboxConfig()
    report = VerificationReport()
    draft = inv.test_draft
    if inv.is_mock:
        reason = "The MOCK provider writes no fix or test, so nothing was run."
        return _finish(report, Decision(NOT_RUN, (reason,)))
    if draft is None or not draft.usable:
        reason = "There is no usable regression-test draft, so nothing was run."
        return _finish(report, Decision(NOT_RUN, (reason,)))
    related = inv.context.related_test_ids()
    ids = [draft.path, *related]
    report.draft_path, report.test_ids = draft.path, ids
    exception_type = inv.traceback.exception_type
    root = Path(repo_root)
    try:
        with Sandbox(root, config) as box:
            report.notes.append(_copy_note(box))
            if box.exists(draft.path):
                reason = f"{draft.path} already exists in the repository, so nothing was run."
                return _finish(report, Decision(INCONCLUSIVE, (reason,)))
            box.write_file(draft.path, draft.code)
            report.before = box.run_pytest(ids)
    except SandboxError as exc:
        return _finish(report, Decision(INCONCLUSIVE, (str(exc),)))

    first = decide(report.before, None, draft.path, related, exception_type)
    fix = inv.fix
    if not first.reproduced or fix is None or not fix.usable:
        return _finish(report, first)
    try:
        with Sandbox(root, config) as box:
            problem = apply_edits(box, fix.edits)
            if problem:
                return _finish(report, Decision(INCONCLUSIVE, (*first.reasons, problem), True))
            box.write_file(draft.path, draft.code)
            report.after = box.run_pytest(ids)
    except SandboxError as exc:
        return _finish(report, Decision(INCONCLUSIVE, (*first.reasons, str(exc)), True))
    final = decide(report.before, report.after, draft.path, related, exception_type)
    return _finish(report, final)


def verification_to_text(report: VerificationReport) -> str:
    lines = [f"VERIFICATION: {report.verdict.upper()}", report.headline, ""]
    lines += [f"  - {reason}" for reason in report.reasons]
    lines += [f"  note: {note}" for note in report.notes]
    for title, run in (("Before the fix", report.before), ("After the fix", report.after)):
        if run is None:
            continue
        summary = run.summary.text if run.summary else "no summary"
        lines.append("")
        lines.append(
            f"{title}: {run.status} | exit code {run.returncode} | {summary} | "
            f"{run.duration:.1f}s | timed out: {run.timed_out}"
        )
        lines += [f"    {outcome}  {test}" for test, outcome in sorted(run.outcomes.items())]
    if report.before is not None and report.before.status != "passed":
        tail = "\n".join(report.before.stdout.splitlines()[-25:])
        lines += ["", "--- real pytest output before the fix (last lines) ---", tail]
    lines += ["", LIMITS_TEXT, SANDBOX_TEXT]
    return "\n".join(lines)