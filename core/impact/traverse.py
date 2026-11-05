"""Traverse the Phase 4 dependency graph outward from changed code.

KNOWN dependents are reached only through CONFIRMED edges. INFERRED dependents need at least
one UNCERTAIN edge, for example a call on an object of unknown type that lists the changed
method as one possible target. A node is never listed as both: known wins. Nothing is
invented: where the graph has no edge there is no dependent, and static analysis cannot see
every dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from core.bugs.mapping import is_test_path
from core.graph.graph import DependencyGraph
from core.graph.models import CALLS, CONFIRMED, IMPORTS, NODE_FILE, Relation
from core.impact.changes import MODULE_LEVEL, ChangedUnit

KNOWN = "known"
INFERRED = "inferred"
MAX_DEPTH_LIMIT = 5


@dataclass(frozen=True)
class TraversalConfig:
    max_depth: int = 4  # hops over call edges (1 = direct callers)
    max_dependents: int = 60  # per changed unit and per status

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= MAX_DEPTH_LIMIT:
            raise ValueError(
                f"IMPACT_MAX_DEPTH must be between 1 and {MAX_DEPTH_LIMIT}, got {self.max_depth}."
            )
        if self.max_dependents < 1:
            raise ValueError("IMPACT_MAX_DEPENDENTS must be at least 1.")


@dataclass(frozen=True)
class Hop:
    """One edge on the way from a dependent toward the changed code."""

    relation: Relation
    to_label: str  # the node this hop leads to
    candidate: bool = False  # True: the hop goes through an uncertain call
    from_module: bool = False  # True: the caller is module-level code of a file

    def text(self) -> str:
        rel = self.relation
        who = f"module-level code in {rel.source_label}" if self.from_module else rel.source_label
        verb = "imports" if rel.kind == IMPORTS else "calls"
        if self.candidate:
            return (
                f"{who} {verb} `{rel.target_label}` at {rel.location} (UNCERTAIN: "
                f"{self.to_label} is one possible target, not confirmed)"
            )
        return f"{who} {verb} {rel.target_label} at {rel.location} ({rel.certainty})"


@dataclass(frozen=True)
class Dependent:
    node_id: str
    label: str
    file: str
    kind: str  # function, method, class, or "module-level"
    status: str  # known or inferred
    depth: int
    path: tuple[Hop, ...]  # from this dependent toward the changed code
    is_test: bool

    @property
    def is_module_level(self) -> bool:
        return self.kind == MODULE_LEVEL

    @property
    def location(self) -> str:
        return self.path[0].relation.location if self.path else ""

    def explain(self) -> str:
        return "; then ".join(hop.text() for hop in self.path)


@dataclass
class Traversal:
    unit_key: str
    dependents: list[Dependent] = field(default_factory=list)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def known(self) -> list[Dependent]:
        return [d for d in self.dependents if d.status == KNOWN and not d.is_test]

    @property
    def inferred(self) -> list[Dependent]:
        return [d for d in self.dependents if d.status == INFERRED and not d.is_test]

    @property
    def known_tests(self) -> list[Dependent]:
        return [d for d in self.dependents if d.status == KNOWN and d.is_test]

    @property
    def inferred_tests(self) -> list[Dependent]:
        return [d for d in self.dependents if d.status == INFERRED and d.is_test]


def _relation(graph: DependencyGraph, source: str, target: str, data: dict) -> Relation:
    return Relation(
        kind=data["kind"],
        certainty=data["certainty"],
        source=source,
        target=target,
        source_label=graph.label(source),
        target_label=graph.label(target),
        file=data["file"],
        line=data["line"],
        reason=data["reason"],
        candidates=tuple(data["candidates"]),
    )


def _in_edges(graph: DependencyGraph, node_id: str, kinds: tuple[str, ...]) -> list[Relation]:
    """Confirmed edges of the given kinds that point at `node_id`, in source order."""
    if not graph.has_node(node_id):
        return []
    found = [
        _relation(graph, source, target, data)
        for source, target, data in graph.nx_graph.in_edges(node_id, data=True)
        if data["kind"] in kinds and data["certainty"] == CONFIRMED
    ]
    found.sort(key=lambda rel: (rel.file, rel.line, rel.source))
    return found


def _is_file(graph: DependencyGraph, node_id: str) -> bool:
    return graph.node(node_id).get("kind") == NODE_FILE


def _make(
    graph: DependencyGraph, node_id: str, hops: tuple[Hop, ...], status: str
) -> Dependent:
    data = graph.node(node_id)
    if data.get("kind") == NODE_FILE:
        file, kind, label = node_id, MODULE_LEVEL, node_id
    else:
        file = str(data.get("file") or "")
        kind = str(data.get("symbol_kind") or data.get("kind") or "")
        label = graph.label(node_id)
    return Dependent(node_id, label, file, kind, status, len(hops), hops, is_test_path(file))


def _known(
    graph: DependencyGraph, start: str, config: TraversalConfig
) -> tuple[list[Dependent], bool]:
    """Callers over confirmed call edges, nearest first. Module-level code is a dead end."""
    found: dict[str, Dependent] = {}
    seen = {start}
    queue: deque[tuple[str, tuple[Hop, ...]]] = deque([(start, ())])
    truncated = False
    while queue:
        node, hops = queue.popleft()
        fresh = [rel for rel in _in_edges(graph, node, (CALLS,)) if rel.source not in seen]
        if not fresh:
            continue
        if len(hops) >= config.max_depth:
            truncated = True
            continue
        for rel in fresh:
            if rel.source in seen:
                continue
            if len(found) >= config.max_dependents:
                return list(found.values()), True
            seen.add(rel.source)
            from_module = _is_file(graph, rel.source)
            path = (Hop(rel, rel.target_label, False, from_module), *hops)
            found[rel.source] = _make(graph, rel.source, path, KNOWN)
            if not from_module:
                queue.append((rel.source, path))
    return list(found.values()), truncated


def _symbol_importers(
    graph: DependencyGraph, start: str, known: list[Dependent]
) -> list[Dependent]:
    """Files that import the changed name but have no known caller of it (for example the
    name is only passed around). Files that already have a known dependent are skipped."""
    covered = {dep.file for dep in known}
    found: dict[str, Dependent] = {}
    for rel in _in_edges(graph, start, (IMPORTS,)):
        source = rel.source
        if source in covered or source in found or not _is_file(graph, source):
            continue
        found[source] = _make(graph, source, (Hop(rel, rel.target_label),), KNOWN)
    return list(found.values())


def _importers(graph: DependencyGraph, file_node: str) -> list[Dependent]:
    """Files that import a module. Their use of the changed code is not known."""
    found: dict[str, Dependent] = {}
    for rel in _in_edges(graph, file_node, (IMPORTS,)):
        source = rel.source
        if source == file_node or source in found or not _is_file(graph, source):
            continue
        found[source] = _make(graph, source, (Hop(rel, rel.target_label),), KNOWN)
    return list(found.values())


def _inferred(
    graph: DependencyGraph, start: str, known: list[Dependent], config: TraversalConfig
) -> list[Dependent]:
    """Dependents whose chain contains at least one uncertain call. Known nodes are skipped."""
    skip = {start, *(dep.node_id for dep in known)}
    found: dict[str, Dependent] = {}
    queue: deque[tuple[str, tuple[Hop, ...]]] = deque()

    def add(rel: Relation, node: str, hops: tuple[Hop, ...], candidate: bool) -> None:
        source = rel.source
        if source in skip or source in found or len(found) >= config.max_dependents:
            return
        from_module = _is_file(graph, source)
        to_label = graph.label(node) if candidate else rel.target_label
        path = (Hop(rel, to_label, candidate, from_module), *hops)
        found[source] = _make(graph, source, path, INFERRED)
        if not from_module:
            queue.append((source, path))

    seeds: list[tuple[str, tuple[Hop, ...]]] = [(start, ())]
    seeds += [(dep.node_id, dep.path) for dep in known if not dep.is_module_level]
    for node, hops in seeds:
        if len(hops) < config.max_depth:
            for rel in graph.possible_callers_of(node):
                add(rel, node, hops, True)
    while queue:
        node, hops = queue.popleft()
        if len(hops) >= config.max_depth:
            continue
        for rel in graph.possible_callers_of(node):
            add(rel, node, hops, True)
        for rel in _in_edges(graph, node, (CALLS,)):
            add(rel, node, hops, False)
    return list(found.values())


def traverse_unit(
    graph: DependencyGraph, unit: ChangedUnit, config: TraversalConfig | None = None
) -> Traversal:
    """Known and inferred dependents of one changed function, class or module."""
    config = config or TraversalConfig()
    result = Traversal(unit.key)
    if not graph.has_node(unit.key):
        result.notes.append(
            f"{unit.qualname} is not in the dependency graph, so nothing could be traced."
        )
        return result
    if unit.is_module_level:
        result.dependents = _importers(graph, unit.key)
        result.notes.append(
            "Module-level code changed. Only files that import this module are known. "
            "Functions in the same file may use the changed names, and the use of module "
            "variables is not tracked."
        )
        return result
    known, truncated = _known(graph, unit.key, config)
    known = [*known, *_symbol_importers(graph, unit.key, known)]
    inferred = _inferred(graph, unit.key, known, config)
    result.dependents = [*known, *inferred]
    result.truncated = truncated
    if truncated:
        result.notes.append(
            f"The trace stopped at depth {config.max_depth} (or at the dependent limit); "
            "more code may be affected further out."
        )
    return result