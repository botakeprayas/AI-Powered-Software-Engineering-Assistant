"""Build the dependency graph from a Phase 1 analysis (same symbol keys as the search index)."""

from __future__ import annotations

import ast
from pathlib import Path

from core.graph.analyzer import (
    CallSite,
    collect_imports,
    collect_locals,
    iter_calls,
    iter_definitions,
    module_assigned_names,
    package_parts_for,
    parse_file,
    short_expr,
    start_line_of,
)
from core.graph.graph import DependencyGraph
from core.graph.models import (
    CALLS,
    CONFIRMED,
    DEFINES,
    IMPORTS,
    NODE_CLASS,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_MODULE,
    NODE_UNRESOLVED,
    UNCERTAIN,
)
from core.graph.resolver import IGNORED, METHOD_KINDS, CallContext, ModuleIndex
from core.repo.pipeline import RepoAnalysis
from core.semantic.models import Symbol


def build_graph(analysis: RepoAnalysis) -> DependencyGraph:
    """Nodes: files, modules, functions, classes. Edges: imports, defines, calls."""
    graph = DependencyGraph()
    root = Path(analysis.root)
    symbols = list(analysis.symbols)
    python_files = sorted(p for p in analysis.file_status if p.endswith(".py"))
    index = ModuleIndex(symbols, python_files)
    trees = _parse_files(root, python_files, analysis.file_status, index)
    _add_nodes(graph, python_files, trees, symbols, index)
    _add_import_edges(graph, trees, index)
    _add_call_edges(graph, trees, symbols, index)
    graph.stats.files_analyzed = len(trees)
    graph.stats.files_skipped = len(python_files) - len(trees)
    return graph


def _parse_files(
    root: Path, python_files: list[str], file_status: dict[str, str], index: ModuleIndex
) -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in python_files:
        if file_status.get(path) != "parsed":
            continue  # Phase 1 already reported why (syntax error, encoding, empty, ...)
        try:
            tree = parse_file(root, path)
        except (OSError, ValueError, SyntaxError, RecursionError):
            continue
        trees[path] = tree
        imports = collect_imports(tree, package_parts_for(path))
        index.register_file(path, imports, module_assigned_names(tree.body))
    return trees


def _add_nodes(
    graph: DependencyGraph,
    python_files: list[str],
    trees: dict[str, ast.Module],
    symbols: list[Symbol],
    index: ModuleIndex,
) -> None:
    for path in python_files:
        graph.add_node(path, NODE_FILE, path, file=path, parsed=path in trees)
    for s in symbols:
        graph.add_node(s.file, NODE_FILE, s.file, file=s.file, parsed=True)
        kind = NODE_CLASS if s.kind == "class" else NODE_FUNCTION
        graph.add_node(
            s.key, kind, s.qualname, file=s.file, line=s.start_line, end_line=s.end_line,
            symbol_kind=s.kind,
        )
    for s in symbols:
        parent = index.parent_of.get(s.key, s.file)
        graph.add_edge(parent, s.key, DEFINES, CONFIRMED, s.file, s.start_line, "defined here")


def _add_import_edges(
    graph: DependencyGraph, trees: dict[str, ast.Module], index: ModuleIndex
) -> None:
    for path in trees:
        for res in index.scope(path).results:
            if res.edge_kind == "none":
                continue
            target = res.edge_target
            if res.edge_kind == "external":
                target = f"module:{res.edge_target}"
                graph.add_node(target, NODE_MODULE, res.edge_target)
            elif res.edge_kind == "unresolved":
                graph.add_node(target, NODE_UNRESOLVED, target.removeprefix("unresolved:"))
            if not graph.has_node(target):
                continue
            graph.add_edge(
                path, target, IMPORTS, res.certainty, path, res.line, res.reason, res.candidates
            )


def _add_call_edges(
    graph: DependencyGraph,
    trees: dict[str, ast.Module],
    symbols: list[Symbol],
    index: ModuleIndex,
) -> None:
    lookup = {(s.file, s.qualname, s.start_line): s for s in symbols}
    for path, tree in trees.items():
        for node, qualname, _in_class in iter_definitions(tree.body):
            symbol = lookup.get((path, qualname, start_line_of(node)))
            if symbol is None:
                continue
            if isinstance(node, ast.ClassDef):
                ctx = CallContext(path)
                sites = iter_calls(node.body, enter_defs=False)
            else:
                class_key = index.parent_of.get(symbol.key) if symbol.kind in METHOD_KINDS else None
                ctx = CallContext(path, collect_locals(node), class_key)
                sites = iter_calls(node.body, enter_defs=True)
            _emit(graph, index, symbol.key, ctx, sites)
        _emit(graph, index, path, CallContext(path), iter_calls(tree.body, enter_defs=False))


def _emit(
    graph: DependencyGraph, index: ModuleIndex, caller: str, ctx: CallContext, sites
) -> None:
    for site in sorted(sites, key=lambda s: s.line):
        res = index.resolve_call(ctx, site)
        note = "decorator: " if site.is_decorator else ""
        if res.status == IGNORED:
            ignored = graph.stats.ignored
            ignored[res.reason] = ignored.get(res.reason, 0) + 1
        elif res.status == CONFIRMED and graph.has_node(res.target):
            graph.add_edge(caller, res.target, CALLS, CONFIRMED, ctx.path, site.line,
                           note + res.reason)
        else:
            _add_uncertain(graph, caller, ctx, site, res, note)


def _add_uncertain(graph, caller: str, ctx: CallContext, site: CallSite, res, note: str) -> None:
    label = res.label or short_expr(site.func)
    node_id = f"unresolved:{label}"
    graph.add_node(node_id, NODE_UNRESOLVED, label)
    reason = note + (res.reason or "target could not be resolved")
    graph.add_edge(caller, node_id, CALLS, UNCERTAIN, ctx.path, site.line, reason, res.candidates)