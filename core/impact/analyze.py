"""Change-impact analysis: changed code -> known and inferred impact -> heuristic ranks.

The ranks are HEURISTIC. They order changed code by transparent factors. They are not
probabilities and have not been calibrated against real defects.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

from core.bugs.mapping import is_test_path
from core.graph.expand import GraphContext
from core.impact.changes import ChangedUnit, ChangeSet, IndexView
from core.impact.rank import (
    HEURISTIC_DISCLAIMER,
    Factor,
    band_for,
    churn_factor,
    dependents_factor,
    heuristic_score,
    lint_factor,
    position_factor,
    size_factor,
    test_factor,
)
from core.impact.signals import CombinedSignals, Finding, GitSignals, RuffSignals
from core.impact.traverse import (
    INFERRED,
    KNOWN,
    Dependent,
    Traversal,
    TraversalConfig,
    traverse_unit,
)
from core.semantic.models import Symbol
from core.settings import DEFAULT_ENV_FILE

STATUS_CHANGED = "changed"
PRIORITY = {STATUS_CHANGED: 0, KNOWN: 1, INFERRED: 2}
NAME_MATCH_DEPTH = 9
MAX_CHURN_DAYS = 3650  # ten years; very long windows are rejected
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS = frozenset({"0", "false", "no", "off"})

LIMITATIONS = (
    "Python only, and only code that the Phase 1 scanner indexed.",
    "Known means every link is a confirmed edge in the dependency graph. It does not prove "
    "that the code runs.",
    "Inferred means at least one link is uncertain (for example a call on an object of "
    "unknown type). An uncertain call lists at most 5 candidate targets.",
    "Calls made through getattr, callbacks, frameworks, threads or dynamic imports are "
    "invisible, so a missing item is not proof of safety.",
    "The use of module-level variables and constants is not tracked, so a changed constant "
    "only reaches the files that import its module.",
    "Test reach is static reachability over call edges, not measured coverage.",
    "Churn is per file, and Ruff findings come from the indexed version with default rules.",
    "Scores are heuristic ranks, not probabilities.",
)


@dataclass(frozen=True)
class ImpactConfig:
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    churn_days: int = 180
    use_git: bool = True
    use_ruff: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.churn_days <= MAX_CHURN_DAYS:
            raise ValueError(f"IMPACT_CHURN_DAYS must be between 1 and {MAX_CHURN_DAYS}.")


def _whole(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number, got {raw!r}.") from exc


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    word = raw.strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be on or off, got {raw!r}.")


def load_impact_config(env_file: Path | None = DEFAULT_ENV_FILE) -> ImpactConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    base = ImpactConfig()
    traversal = TraversalConfig(
        max_depth=_whole("IMPACT_MAX_DEPTH", base.traversal.max_depth),
        max_dependents=_whole("IMPACT_MAX_DEPENDENTS", base.traversal.max_dependents),
    )
    return ImpactConfig(
        traversal,
        _whole("IMPACT_CHURN_DAYS", base.churn_days),
        _flag("IMPACT_USE_GIT", True),
        _flag("IMPACT_USE_RUFF", True),
    )


def default_signals(root: Path, config: ImpactConfig) -> CombinedSignals:
    git = GitSignals(root, config.churn_days) if config.use_git else None
    ruff = RuffSignals(root) if config.use_ruff else None
    return CombinedSignals(git, ruff)


@dataclass
class UnitImpact:
    unit: ChangedUnit
    traversal: Traversal
    factors: list[Factor]
    score: int
    band: str
    public: bool
    signature_changed: bool
    findings: list[Finding]
    churn: int | None
    notes: list[str] = field(default_factory=list)
    rank: int = 0

    @property
    def known(self) -> list[Dependent]:
        return self.traversal.known

    @property
    def inferred(self) -> list[Dependent]:
        return self.traversal.inferred


@dataclass(frozen=True)
class AffectedFile:
    path: str
    status: str  # known, or inferred (no known dependent in this file)
    labels: tuple[str, ...]
    reasons: tuple[str, ...]
    min_depth: int
    also_changed: bool  # this file is also edited by the diff


@dataclass(frozen=True)
class RecommendedTest:
    id: str  # a pytest node id, or a test file
    status: str  # changed, known or inferred
    reason: str
    file: str
    depth: int = 0


@dataclass
class ImpactReport:
    changes: ChangeSet
    units: list[UnitImpact]
    changed_files: list[str]
    affected_known: list[AffectedFile]
    affected_inferred: list[AffectedFile]
    tests: list[RecommendedTest]
    skipped: list[tuple[str, str, str]]  # (path, outcome, detail) for files not analyzed
    notes: list[str]
    disclaimer: str = HEURISTIC_DISCLAIMER
    limitations: tuple[str, ...] = LIMITATIONS

    @property
    def known_file_paths(self) -> list[str]:
        return sorted(f.path for f in self.affected_known)

    @property
    def inferred_file_paths(self) -> list[str]:
        return sorted(f.path for f in self.affected_inferred)

    @property
    def known_test_ids(self) -> list[str]:
        return [t.id for t in self.tests if t.status == KNOWN]

    @property
    def inferred_test_ids(self) -> list[str]:
        return [t.id for t in self.tests if t.status == INFERRED]

    @property
    def test_command(self) -> str:
        ids = [f'"{t.id}"' if " " in t.id else t.id for t in self.tests]
        return ("pytest " + " ".join(ids)) if ids else ""


def pytest_id(dep: Dependent) -> str:
    if dep.is_module_level:
        return dep.file
    return f"{dep.file}::{dep.label.replace('.', '::')}"


def _is_public(qualname: str) -> bool:
    for part in qualname.split("."):
        if part.startswith("_") and not (part.startswith("__") and part.endswith("__")):
            return False
    return True


def _signature_line(symbol: Symbol | None) -> int | None:
    if symbol is None:
        return None
    for offset, text in enumerate(symbol.code.split("\n")):
        if text.lstrip().startswith(("def ", "async def ", "class ")):
            return symbol.start_line + offset
    return None


def _findings_for(
    unit: ChangedUnit, per_file: dict[str, list[Finding]] | None, view: IndexView
) -> list[Finding] | None:
    if per_file is None:
        return None
    found = per_file.get(unit.file, [])
    if not unit.is_module_level:
        return [f for f in found if unit.start_line <= f.line <= unit.end_line]
    spans = [(s.start_line, s.end_line) for s in view.symbols_in(unit.file)]
    return [f for f in found if not any(a <= f.line <= b for a, b in spans)]


def _unit_notes(unit: ChangedUnit, impact: UnitImpact) -> list[str]:
    notes = list(impact.traversal.notes)
    if unit.whole:
        notes.append(
            f"The whole file is {unit.whole}: every importer fails at import time "
            "(confirmed import edges)."
        )
    if impact.signature_changed and not unit.whole:
        notes.append("The definition line is edited: callers may need to change as well.")
    if not impact.known and impact.inferred:
        notes.append(
            "No confirmed dependents; the inferred ones come from calls on objects of "
            "unknown type."
        )
    if impact.public and not unit.is_module_level and not impact.known:
        notes.append(
            "Public name with no confirmed dependents: code outside this repository, or "
            "dynamic callers, may still use it. Static analysis cannot see that."
        )
    if not impact.traversal.known_tests:
        notes.append("No test reaches this code through confirmed edges.")
    return notes


def _affected_files(
    impacts: list[UnitImpact], changed_paths: set[str], changed_keys: set[str]
) -> tuple[list[AffectedFile], list[AffectedFile]]:
    buckets: dict[str, dict[str, list[Dependent]]] = {KNOWN: {}, INFERRED: {}}
    for impact in impacts:
        for dep in impact.traversal.dependents:
            if dep.is_test or dep.node_id in changed_keys:
                continue
            buckets[dep.status].setdefault(dep.file, []).append(dep)

    def build(status: str, skip: set[str]) -> list[AffectedFile]:
        files = []
        for path, deps in sorted(buckets[status].items()):
            if path in skip:
                continue
            ordered = sorted(deps, key=lambda d: (d.depth, d.label))
            labels = tuple(dict.fromkeys(d.label for d in ordered))
            reasons = tuple(dict.fromkeys(d.explain() for d in ordered))[:3]
            files.append(
                AffectedFile(
                    path, status, labels, reasons, ordered[0].depth, path in changed_paths
                )
            )
        return files

    known = build(KNOWN, set())
    inferred = build(INFERRED, {f.path for f in known})
    return known, inferred


def _recommend(
    impacts: list[UnitImpact], view: IndexView, units: list[ChangedUnit]
) -> list[RecommendedTest]:
    found: dict[str, RecommendedTest] = {}

    def put(item: RecommendedTest) -> None:
        current = found.get(item.id)
        if current is None or PRIORITY[item.status] < PRIORITY[current.status]:
            found[item.id] = item

    for unit in units:
        short = unit.qualname.rsplit(".", 1)[-1]
        if is_test_path(unit.file) and not unit.is_module_level and short.startswith("test"):
            test_id = f"{unit.file}::{unit.qualname.replace('.', '::')}"
            reason = "this test is edited by the diff"
            put(RecommendedTest(test_id, STATUS_CHANGED, reason, unit.file))
    for impact in impacts:
        name = impact.unit.qualname
        for status, deps in ((KNOWN, impact.traversal.known_tests),
                             (INFERRED, impact.traversal.inferred_tests)):
            for dep in deps:
                put(RecommendedTest(
                    pytest_id(dep), status, f"reaches {name}: {dep.explain()}", dep.file, dep.depth
                ))
    for path in sorted({u.file for u in units}):
        stem = PurePosixPath(path).stem
        for test_file in sorted(view.files):
            if not is_test_path(test_file):
                continue
            if PurePosixPath(test_file).stem not in (f"test_{stem}", f"{stem}_test"):
                continue
            if any(t == test_file or t.startswith(test_file + "::") for t in found):
                continue
            put(RecommendedTest(
                test_file, INFERRED, f"named after {path} (name match only; no graph link)",
                test_file, NAME_MATCH_DEPTH,
            ))
    for test_id in list(found):  # a specific test beats the bare file of the same tests
        if "::" not in test_id and any(other.startswith(test_id + "::") for other in found):
            del found[test_id]
    return sorted(found.values(), key=lambda t: (PRIORITY[t.status], t.depth, t.id))


def analyze_impact(
    changes: ChangeSet,
    context: GraphContext,
    view: IndexView,
    config: ImpactConfig | None = None,
    signals=None,
) -> ImpactReport:
    """Rank the changed code and report what it may affect. Never raises for odd input."""
    config = config or ImpactConfig()
    signals = signals if signals is not None else default_signals(context.root, config)
    notes = list(changes.warnings)
    skipped = [(c.path, c.outcome, c.detail) for c in changes.files if not c.units]
    notes += [f"{c.path}: {c.detail}" for c in changes.files if c.units and c.outcome == "unparsed"]
    units = changes.units
    if not units:
        notes.append("No changed Python code could be mapped, so no impact was computed.")
        return ImpactReport(changes, [], [], [], [], [], skipped, notes)

    changed_keys = {u.key for u in units}
    changed_paths = sorted({u.file for u in units})
    lint_paths = [p for p in changed_paths if p.endswith(".py") and view.files.get(p, False)]
    per_file = signals.lint(lint_paths) if lint_paths else {}
    if per_file is None:
        notes.append("Ruff was not available, so static-analysis findings are not scored.")
    churn_by_file = {path: signals.churn(path) for path in changed_paths}
    if any(value is None for value in churn_by_file.values()):
        notes.append("Git history was not available, so churn is not scored.")

    impacts: list[UnitImpact] = []
    for unit in units:
        trav = traverse_unit(context.graph, unit, config.traversal)
        known = [d for d in trav.known if d.node_id not in changed_keys]
        inferred = [d for d in trav.inferred if d.node_id not in changed_keys]
        findings = _findings_for(unit, per_file, view)
        churn = churn_by_file[unit.file]
        symbol = context.symbols.get(unit.key)
        signature = _signature_line(symbol)
        edited_definition = signature is not None and signature in unit.touched
        changed_signature = bool(unit.whole) or edited_definition
        factors = [
            dependents_factor(len(known), len(inferred)),
            position_factor(
                max((d.depth for d in known), default=0),
                any(d.is_module_level for d in known),
                trav.truncated,
                config.traversal.max_depth,
            ),
            test_factor(len(trav.known_tests)),
            churn_factor(churn, config.churn_days),
            lint_factor(None if findings is None else [f.code for f in findings]),
            size_factor(unit.lines_changed, bool(unit.whole)),
        ]
        score = heuristic_score(factors)
        impact = UnitImpact(
            unit, trav, factors, score, band_for(score),
            not unit.is_module_level and _is_public(unit.qualname), changed_signature,
            findings or [], churn,
        )
        impact.notes = _unit_notes(unit, impact)
        impacts.append(impact)
    impacts.sort(key=lambda i: (-i.score, i.unit.file, i.unit.start_line, i.unit.key))
    for position, impact in enumerate(impacts, start=1):
        impact.rank = position

    known_files, inferred_files = _affected_files(impacts, set(changed_paths), changed_keys)
    notes += [n for n in signals.notes if n not in notes]
    return ImpactReport(
        changes, impacts, changed_paths, known_files, inferred_files,
        _recommend(impacts, view, units), skipped, list(dict.fromkeys(notes)),
    )


@dataclass(frozen=True)
class Comparison:
    """Predictions measured against a hand-checked list. This checks the traversal, not the
    heuristic score."""

    expected: tuple[str, ...]
    found_known: tuple[str, ...]
    found_inferred_only: tuple[str, ...]
    missed: tuple[str, ...]
    extra_known: tuple[str, ...]
    extra_inferred: tuple[str, ...]

    @property
    def recall_known(self) -> float | None:
        return len(self.found_known) / len(self.expected) if self.expected else None

    @property
    def recall_any(self) -> float | None:
        hits = len(self.found_known) + len(self.found_inferred_only)
        return hits / len(self.expected) if self.expected else None

    @property
    def precision_known(self) -> float | None:
        total = len(self.found_known) + len(self.extra_known)
        return len(self.found_known) / total if total else None

    def summary(self) -> str:
        def rate(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.0%}"

        return (
            f"{len(self.found_known)} of {len(self.expected)} expected items were found as "
            f"known (recall {rate(self.recall_known)}), {len(self.found_inferred_only)} more "
            f"only as inferred, {len(self.missed)} missed. Known precision: "
            f"{rate(self.precision_known)} ({len(self.extra_known)} known items were not "
            "on your list)."
        )


def compare_sets(
    known: Iterable[str], inferred: Iterable[str], expected: Iterable[str]
) -> Comparison:
    wanted = set(expected)
    known_set = set(known)
    inferred_set = set(inferred) - known_set
    return Comparison(
        tuple(sorted(wanted)),
        tuple(sorted(wanted & known_set)),
        tuple(sorted(wanted & inferred_set)),
        tuple(sorted(wanted - known_set - inferred_set)),
        tuple(sorted(known_set - wanted)),
        tuple(sorted(inferred_set - wanted)),
    )


def compare_files(report: ImpactReport, expected: Iterable[str]) -> Comparison:
    """Affected files (tests excluded) against a hand-checked list of affected files."""
    return compare_sets(report.known_file_paths, report.inferred_file_paths, expected)


def compare_tests(report: ImpactReport, expected: Iterable[str]) -> Comparison:
    """Recommended tests (those edited by the diff excluded) against a hand-checked list."""
    return compare_sets(report.known_test_ids, report.inferred_test_ids, expected)