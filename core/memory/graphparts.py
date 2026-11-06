"""Per-file dependency-graph data for the memory store (Phase 8).

A file's edges are computed with the Phase 4 builder code itself, so a graph assembled from
stored per-file pieces equals a graph built from scratch (the tests check this). A file's
graph data depends on its own source AND on the interfaces of other files (their
definitions, imports and module variables), so stored pieces may be reused only while no
interface has changed. The indexer applies that rule.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from core.graph.analyzer import ImportStmt
from core.graph.builder import _add_call_edges, _add_import_edges, _add_nodes
from core.graph.graph import DependencyGraph
from core.graph.models import NODE_CLASS, NODE_FILE, NODE_FUNCTION
from core.graph.resolver import ModuleIndex
from core.semantic.models import Symbol

EXTRA_PREFIXES = ("module:", "unresolved:")
OWNED_KINDS = (NODE_FILE, NODE_FUNCTION, NODE_CLASS)


class MemoryIndexError(RuntimeError):
    """The stored index data is incomplete or inconsistent."""


@dataclass
class FileGraph:
    """The graph data one file contributes: its nodes, its edges and ignored-call counts."""

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    ignored: dict[str, int] = field(default_factory=dict)


def imports_to_json(imports: Sequence[ImportStmt]) -> str:
    return json.dumps(
        [
            [
                s.line, s.module, [list(pair) for pair in s.names], s.alias, s.is_from,
                s.star, s.unresolved_relative,
            ]
            for s in imports
        ]
    )


def imports_from_json(text: str) -> list[ImportStmt]:
    return [
        ImportStmt(
            line, module, tuple((n, a) for n, a in names), alias, is_from, star, relative
        )
        for line, module, names, alias, is_from, star, relative in json.loads(text)
    ]


def interface_hash(
    status: str,
    symbols: Sequence[Symbol],
    imports: Sequence[ImportStmt] | None,
    assigned: Iterable[str] | None,
) -> str:
    """A hash of everything OTHER files can see of this file.

    That is: its parse status, its definitions (key, kind, name, qualname and which symbol
    contains which), its imports (without line numbers) and its module-level variable names.
    Function bodies, docstrings, parameters and line numbers are not part of it, because no
    other file's resolution depends on them. `imports` is None for a file that was not
    parsed for the graph.
    """
    parents = ModuleIndex(list(symbols), []).parent_of
    imported = None
    if imports is not None:
        imported = [
            [s.module, [list(p) for p in s.names], s.alias, s.is_from, s.star,
             s.unresolved_relative]
            for s in imports
        ]
    payload = {
        "status": status,
        "symbols": [[s.key, s.kind, s.name, s.qualname, parents.get(s.key)] for s in symbols],
        "imports": imported,
        "assigned": None if assigned is None else sorted(assigned),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_graphs(
    symbols: Sequence[Symbol],
    python_files: Sequence[str],
    imports: Mapping[str, Sequence[ImportStmt]],
    assigned: Mapping[str, Iterable[str]],
    trees: Mapping[str, ast.Module],
    targets: Iterable[str],
) -> dict[str, FileGraph]:
    """Graph data for the `targets` files, computed by the Phase 4 builder code.

    `symbols` and `python_files` cover the WHOLE repository (names resolve across files).
    `imports` and `assigned` hold every file that was parsed for the graph. `trees` holds
    the parsed source of the targets only.
    """
    wanted = sorted(set(targets))
    all_symbols = list(symbols)
    index = ModuleIndex(all_symbols, list(python_files))
    for path, statements in imports.items():
        index.register_file(path, list(statements), set(assigned.get(path, ())))
    graph = DependencyGraph()
    # The skeleton: every file and symbol node, so that edges can point anywhere.
    _add_nodes(graph, list(python_files), {path: None for path in imports}, all_symbols, index)
    by_file: dict[str, list[Symbol]] = {}
    for symbol in all_symbols:
        by_file.setdefault(symbol.file, []).append(symbol)
    ignored: dict[str, dict[str, int]] = {}
    for path in wanted:
        tree = trees.get(path)
        if tree is None:
            continue
        before = dict(graph.stats.ignored)
        _add_import_edges(graph, {path: tree}, index)
        _add_call_edges(graph, {path: tree}, by_file.get(path, []), index)
        ignored[path] = {
            key: count - before.get(key, 0)
            for key, count in graph.stats.ignored.items()
            if count - before.get(key, 0) > 0
        }
    return _extract(graph, wanted, ignored)


def _extract(
    graph: DependencyGraph, wanted: list[str], ignored: dict[str, dict[str, int]]
) -> dict[str, FileGraph]:
    nx_graph = graph.nx_graph
    wanted_set = set(wanted)
    result = {path: FileGraph(ignored=dict(ignored.get(path, {}))) for path in wanted}
    seen: dict[str, set[str]] = {path: set() for path in wanted}

    def add_node(path: str, node_id: str, data: dict) -> None:
        if node_id in seen[path]:
            return
        seen[path].add(node_id)
        attrs = {k: v for k, v in data.items() if k not in ("kind", "label")}
        result[path].nodes.append(
            {"id": node_id, "kind": data["kind"], "label": data["label"], "attrs": attrs}
        )

    for node_id, data in nx_graph.nodes(data=True):
        owner = data.get("file")
        if data.get("kind") in OWNED_KINDS and owner in wanted_set:
            add_node(owner, node_id, data)
    for source, target, data in nx_graph.edges(data=True):
        owner = data["file"]
        if owner not in wanted_set:
            continue
        result[owner].edges.append(
            {
                "source": source, "target": target, "kind": data["kind"],
                "certainty": data["certainty"], "file": owner, "line": data["line"],
                "reason": data["reason"], "candidates": list(data["candidates"]),
            }
        )
        if target.startswith(EXTRA_PREFIXES):
            add_node(owner, target, dict(nx_graph.nodes[target]))
    return result


def assemble_graph(
    parts: Mapping[str, FileGraph], files_analyzed: int, files_skipped: int
) -> DependencyGraph:
    """Put the stored per-file pieces together. Raises KeyError if an edge has no node."""
    graph = DependencyGraph()
    for path in sorted(parts):
        for node in parts[path].nodes:
            graph.add_node(node["id"], node["kind"], node["label"], **node["attrs"])
    for path in sorted(parts):
        for edge in parts[path].edges:
            graph.add_edge(
                edge["source"], edge["target"], edge["kind"], edge["certainty"],
                edge["file"], edge["line"], edge["reason"], tuple(edge["candidates"]),
            )
    for path in sorted(parts):
        for reason, count in parts[path].ignored.items():
            graph.stats.ignored[reason] = graph.stats.ignored.get(reason, 0) + count
    graph.stats.files_analyzed = files_analyzed
    graph.stats.files_skipped = files_skipped
    return graph