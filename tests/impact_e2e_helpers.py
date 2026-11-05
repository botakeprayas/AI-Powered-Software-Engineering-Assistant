"""Helpers for the Phase 7 analysis tests on data/impact_sample."""

from __future__ import annotations

from functools import lru_cache

from impact_helpers import IMPACT_SAMPLE, make_diff, replace_line

from core.graph.expand import GraphContext, load_graph_context
from core.impact.analyze import ImpactConfig, ImpactReport, analyze_impact
from core.impact.changes import IndexView, map_diff
from core.impact.diff_parser import parse_diff
from core.impact.signals import Finding


@lru_cache(maxsize=1)
def sample_context() -> GraphContext:
    return load_graph_context(IMPACT_SAMPLE)


def sample_text(path: str) -> str:
    return (IMPACT_SAMPLE / path).read_text(encoding="utf-8")


def edit_diff(path: str, line: int, new: str) -> str:
    old = sample_text(path)
    return make_diff(path, old, replace_line(old, line, new))


class FakeSignals:
    """MOCK signals with fixed churn and findings. Not real git or Ruff output."""

    def __init__(self, churn=None, findings=None, git=True, ruff=True):
        self._churn = churn or {}
        self._findings = findings or {}
        self.git = git
        self.ruff = ruff
        self.notes: list[str] = []

    def churn(self, path):
        return self._churn.get(path, 0) if self.git else None

    def lint(self, paths):
        if not self.ruff:
            return None
        return {path: list(self._findings.get(path, [])) for path in paths}


def unused_variable(path: str = "pricing/rates.py", line: int = 13) -> Finding:
    return Finding(path, line, line, "F841", "Local variable `unused` is assigned but never used")


def analyze_diff(diff: str, signals=None, config=None, base: str = "auto") -> ImpactReport:
    context = sample_context()
    view = IndexView.from_context(context)
    changes = map_diff(parse_diff(diff), view, base)
    return analyze_impact(
        changes, context, view, config or ImpactConfig(),
        signals if signals is not None else FakeSignals(),
    )