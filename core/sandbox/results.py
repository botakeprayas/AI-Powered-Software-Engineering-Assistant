"""Pytest output parsing and the result of one sandboxed run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_NO_TESTS = "no_tests"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_INCONCLUSIVE = "inconclusive"

COUNT_KEYS = {
    "passed": "passed", "failed": "failed", "error": "errors", "errors": "errors",
    "skipped": "skipped", "xfailed": "xfailed", "xpassed": "xpassed",
    "deselected": "deselected", "warning": "warnings", "warnings": "warnings",
}
COUNT_RE = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)"
)
TIME_RE = re.compile(r"\bin (\d+(?:\.\d+)?)s\b")
OUTCOME_RE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(\S+)")
SUMMARY_HEADER = "short test summary info"


@dataclass(frozen=True)
class PytestSummary:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0
    warnings: int = 0
    seconds: float | None = None
    no_tests: bool = False

    @property
    def text(self) -> str:
        parts = [f"{n} {name}" for name, n in (
            ("passed", self.passed), ("failed", self.failed), ("errors", self.errors),
            ("skipped", self.skipped),
        ) if n]
        return ", ".join(parts) or "no tests ran"


def parse_summary(text: str) -> PytestSummary | None:
    """Read pytest's last line (for example '1 failed, 2 passed in 0.12s')."""
    for line in reversed(text.strip().splitlines()[-12:]):
        cleaned = line.strip().strip("=").strip()
        time_match = TIME_RE.search(cleaned)
        if time_match is None:
            continue
        found = COUNT_RE.findall(cleaned)
        no_tests = cleaned.startswith("no tests ran")
        if not found and not no_tests:
            continue
        counts: dict[str, int] = {}
        for number, word in found:
            key = COUNT_KEYS[word]
            counts[key] = counts.get(key, 0) + int(number)
        return PytestSummary(
            **counts, seconds=float(time_match.group(1)), no_tests=no_tests
        )
    return None


def parse_outcomes(text: str) -> dict[str, str]:
    """Per-test outcomes from the last 'short test summary info' section."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if SUMMARY_HEADER in line]
    if not starts:
        return {}
    outcomes: dict[str, str] = {}
    for line in lines[starts[-1] + 1 :]:
        match = OUTCOME_RE.match(line)
        if match:
            outcomes.setdefault(match.group(2), match.group(1))
    return outcomes


@dataclass
class RunResult:
    command: list[str]
    returncode: int | None
    timed_out: bool
    duration: float
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    summary: PytestSummary | None = None
    outcomes: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def status(self) -> str:
        """passed only when pytest exited 0, printed a summary, and nothing failed."""
        if self.error:
            return STATUS_ERROR
        if self.timed_out:
            return STATUS_TIMEOUT
        summary = self.summary
        if self.returncode == 5 or (summary is not None and summary.no_tests):
            return STATUS_NO_TESTS
        if summary is None:
            return STATUS_INCONCLUSIVE if self.returncode == 0 else STATUS_ERROR
        bad = summary.failed + summary.errors
        if self.returncode == 0 and summary.passed > 0 and bad == 0:
            return STATUS_PASSED
        if self.returncode == 1 and bad > 0:
            return STATUS_FAILED
        if self.returncode in (2, 3, 4):
            return STATUS_ERROR
        return STATUS_INCONCLUSIVE

    def ids_with(self, outcome: str) -> list[str]:
        return sorted(test for test, value in self.outcomes.items() if value == outcome)