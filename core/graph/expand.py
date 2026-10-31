"""Graph-expanded retrieval: add CONFIRMED callers and callees of retrieved code.

Expansion never creates evidence on its own: it starts only from excerpts that already
passed the Phase 3 relevance gate. Uncertain relationships are never followed; they become
notes that the model must report as unknowns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from core.chat.models import Evidence, QAConfig
from core.chat.qa import TRUNCATION_MARKER
from core.graph.builder import build_graph
from core.graph.graph import DependencyGraph
from core.graph.models import NODE_FILE, Relation
from core.repo.pipeline import analyze_repository
from core.semantic.models import Symbol
from core.settings import DEFAULT_ENV_FILE

CALLEE = "callee"
CALLER = "caller"
MAX_DEPTH = 3


@dataclass(frozen=True)
class ExpansionConfig:
    depth: int = 1  # hops over confirmed call edges
    max_per_direction: int = 2  # per seed, per direction
    max_added: int = 4  # added excerpts in total
    max_seeds: int = 3  # retrieved excerpts that are expanded
    max_total_chars: int = 11000  # all excerpts together (seeds included)
    max_notes: int = 8
    include_callees: bool = True
    include_callers: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.depth <= MAX_DEPTH:
            raise ValueError(
                f"GRAPH_EXPAND_DEPTH must be between 1 and {MAX_DEPTH}, got {self.depth}."
            )
        for name in ("max_per_direction", "max_added", "max_seeds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")


def load_expansion_config(env_file: Path | None = DEFAULT_ENV_FILE) -> ExpansionConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    base = ExpansionConfig()

    def whole(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a whole number, got {raw!r}.") from exc

    return ExpansionConfig(
        depth=whole("GRAPH_EXPAND_DEPTH", base.depth),
        max_per_direction=whole("GRAPH_EXPAND_MAX_PER_DIRECTION", base.max_per_direction),
        max_added=whole("GRAPH_EXPAND_MAX_ADDED", base.max_added),
        max_seeds=whole("GRAPH_EXPAND_MAX_SEEDS", base.max_seeds),
    )


@dataclass(frozen=True)
class GraphEvidence(Evidence):
    """An excerpt added because the call graph links it to a retrieved excerpt."""

    via: str = ""  # why it was added, with the source line of every hop
    seed_id: str = ""
    direction: str = ""  # callee or caller
    depth: int = 1
    relations: tuple[Relation, ...] = ()  # the confirmed edges that justify it


@dataclass
class ExpansionResult:
    evidence: list[GraphEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # uncertain / module-level, not citable
    skipped: list[str] = field(default_factory=list)  # why something was not added


@dataclass
class GraphContext:
    graph: DependencyGraph
    symbols: dict[str, Symbol]  # same keys as the graph and the search index
    root: Path


def load_graph_context(root: Path) -> GraphContext:
    """Analyze a folder with Phase 1, then build the dependency graph from the result."""
    analysis = analyze_repository(Path(root))
    try:
        graph = build_graph(analysis)
        symbols = {s.key: s for s in analysis.symbols}
        return GraphContext(graph, symbols, analysis.root)
    finally:
        analysis.close()


def _neighbors(
    graph: DependencyGraph, start: str, direction: str, depth: int
) -> list[tuple[str, int, tuple[Relation, ...]]]:
    """Breadth-first over CONFIRMED call edges. Nearest first; each node appears once."""
    found: list[tuple[str, int, tuple[Relation, ...]]] = []
    seen = {start}
    frontier: list[tuple[str, tuple[Relation, ...]]] = [(start, ())]
    for hop in range(1, depth + 1):
        following: list[tuple[str, tuple[Relation, ...]]] = []
        for current, path in frontier:
            if direction == CALLEE:
                relations = graph.callees_of(current)
            else:
                relations = graph.callers_of(current)
            for rel in relations:
                other = rel.target if direction == CALLEE else rel.source
                if other in seen:
                    continue
                seen.add(other)
                new_path = (*path, rel)
                found.append((other, hop, new_path))
                following.append((other, new_path))
        frontier = following
    return found


def _hop(rel: Relation) -> str:
    return f"{rel.source_label} calls {rel.target_label} at {rel.location}"


def _via(seed: Evidence, direction: str, path: tuple[Relation, ...]) -> str:
    word = "callee" if direction == CALLEE else "caller"
    chain = "; ".join(_hop(rel) for rel in path)
    return f"{word} of {seed.id} (confirmed: {chain})"


def _clip(code: str, limit: int) -> tuple[str, bool]:
    if len(code) <= limit:
        return code, False
    return code[:limit].rstrip() + TRUNCATION_MARKER, True


def _candidates(
    context: GraphContext, seed: Evidence, direction: str, config: ExpansionConfig,
    have: set[str],
) -> tuple[list[tuple[Symbol, int, tuple[Relation, ...]]], list[str]]:
    graph = context.graph
    picked: list[tuple[Symbol, int, tuple[Relation, ...]]] = []
    notes: list[str] = []
    for node_id, hop, path in _neighbors(graph, seed.symbol.key, direction, config.depth):
        if graph.node(node_id).get("kind") == NODE_FILE:
            last = path[-1]  # module-level code has no function to show, so it becomes a note
            name = seed.symbol.qualname
            notes.append(
                f"{name} is also called from module-level code in {last.file}:{last.line} "
                "(confirmed)"
            )
            continue
        symbol = context.symbols.get(node_id)
        if symbol is None or node_id in have:
            continue
        if len(picked) < config.max_per_direction:
            picked.append((symbol, hop, path))
    return picked, notes


def _uncertain_notes(graph: DependencyGraph, seed: Evidence) -> list[str]:
    key, name = seed.symbol.key, seed.symbol.qualname
    notes = []
    for rel in graph.uncertain_callees_of(key)[:2]:
        text = f"{name} calls `{rel.target_label}` at {rel.location}: UNCERTAIN ({rel.reason})"
        if rel.candidates:
            names = ", ".join(graph.label(c) for c in rel.candidates)
            text += f"; possible targets, NOT confirmed: {names}"
        notes.append(text)
    for rel in graph.possible_callers_of(key)[:2]:
        notes.append(
            f"{rel.source_label} might call {name} at {rel.location}: "
            f"UNCERTAIN ({rel.reason}); NOT confirmed"
        )
    return notes


def expand_evidence(
    context: GraphContext,
    seeds: list[Evidence],
    qa_config: QAConfig,
    config: ExpansionConfig | None = None,
) -> ExpansionResult:
    """Add confirmed callers/callees of the seeds, within the configured limits."""
    config = config or ExpansionConfig()
    result = ExpansionResult()
    graph = context.graph
    have = {e.symbol.key for e in seeds}
    directions = []
    if config.include_callees:
        directions.append(CALLEE)
    if config.include_callers:
        directions.append(CALLER)
    pools: list[tuple[Evidence, str, list[tuple[Symbol, int, tuple[Relation, ...]]]]] = []
    notes: list[str] = []
    for seed in seeds[: config.max_seeds]:
        if not graph.has_node(seed.symbol.key):
            name = seed.symbol.qualname
            result.skipped.append(f"{seed.id} ({name}) is not in the dependency graph")
            continue
        for direction in directions:
            picked, extra = _candidates(context, seed, direction, config, have)
            pools.append((seed, direction, picked))
            notes.extend(extra)
        notes.extend(_uncertain_notes(graph, seed))
    result.notes = list(dict.fromkeys(notes))[: config.max_notes]

    used = sum(len(e.code) for e in seeds)
    included = set(have)
    rank = 0
    while len(result.evidence) < config.max_added and any(rank < len(p) for _, _, p in pools):
        for seed, direction, picked in pools:  # round-robin: every seed and direction in turn
            if rank >= len(picked) or len(result.evidence) >= config.max_added:
                continue
            symbol, hop, path = picked[rank]
            if symbol.key in included:
                continue
            code, truncated = _clip(symbol.code, qa_config.max_code_chars)
            if used + len(code) > config.max_total_chars:
                result.skipped.append(f"{symbol.qualname} not added: prompt size limit reached")
                continue
            used += len(code)
            included.add(symbol.key)
            result.evidence.append(
                GraphEvidence(
                    id=f"E{len(seeds) + len(result.evidence) + 1}",
                    symbol=symbol,
                    code=code,
                    code_truncated=truncated,
                    semantic_score=None,
                    keyword_rank=None,
                    keyword_score=None,
                    via=_via(seed, direction, path),
                    seed_id=seed.id,
                    direction=direction,
                    depth=hop,
                    relations=path,
                )
            )
        rank += 1
    left = [s for _, _, p in pools for s, _, _ in p if s.key not in included]
    if left and len(result.evidence) >= config.max_added:
        result.skipped.append(
            "more connected code exists, but the limit of "
            f"{config.max_added} added excerpt(s) was reached"
        )
    result.skipped = list(dict.fromkeys(result.skipped))
    return result