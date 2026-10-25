"""Inspect the dependency graph of a repository.

Examples (from the project root):
  python scripts\\graph.py data\\graph_sample
  python scripts\\graph.py data\\graph_sample --callees OrderService.checkout
  python scripts\\graph.py data\\graph_sample --callers charge
  python scripts\\graph.py data\\graph_sample --uncertain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.graph.builder import build_graph  # noqa: E402
from core.graph.graph import DependencyGraph  # noqa: E402
from core.graph.models import LIMITATIONS, NODE_FILE, Relation  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.pipeline import analyze_repository  # noqa: E402
from core.settings import load_settings  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency graph viewer")
    parser.add_argument("folder", help="repository folder")
    parser.add_argument("--callers", default=None, help="function/class name to look up")
    parser.add_argument("--callees", default=None, help="function/class name to look up")
    parser.add_argument("--uncertain", action="store_true", help="list uncertain relationships")
    parser.add_argument("--edges", action="store_true", help="list every edge with file:line")
    parser.add_argument("--limitations", action="store_true", help="what cannot be resolved")
    return parser.parse_args(argv)


def where(graph: DependencyGraph, node_id: str) -> str:
    data = graph.node(node_id)
    if data.get("kind") == NODE_FILE:
        return node_id
    if data.get("line"):
        return f"{data['file']}:{data['line']}-{data['end_line']}"
    return "unresolved"


def show(graph: DependencyGraph, relations: list[Relation], use_source: bool) -> None:
    if not relations:
        print("  (none)")
        return
    for r in relations:
        node = r.source if use_source else r.target
        label = r.source_label if use_source else r.target_label
        possible = ""
        if r.candidates:
            names = ", ".join(graph.label(c) for c in r.candidates)
            possible = f" | possible targets (not confirmed): {names}"
        print(f"  {label}  [{where(graph, node)}]  at {r.location} | {r.reason}{possible}")


def print_summary(graph: DependencyGraph, root: Path) -> None:
    stats = graph.stats
    print(f"Scanned: {root}")
    print(f"Files analyzed: {stats.files_analyzed} (not parsed: {stats.files_skipped})")
    print("Nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.nodes.items())))
    print("Edges: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.edges.items())))
    ignored = ", ".join(f"{k}={v}" for k, v in sorted(stats.ignored.items())) or "none"
    print(f"Calls not recorded (outside the repo or not analyzable): {ignored}")


def lookup(graph: DependencyGraph, name: str, callers: bool) -> int:
    ids = graph.find_symbols(name)
    if not ids:
        print(f"ERROR: no function or class matches {name!r}", file=sys.stderr)
        return 1
    for node_id in ids:
        print(f"\n{graph.label(node_id)}  [{where(graph, node_id)}]")
        if callers:
            print("Callers (confirmed):")
            show(graph, graph.callers_of(node_id), use_source=True)
            print("Possible callers (uncertain, NOT confirmed):")
            show(graph, graph.possible_callers_of(node_id), use_source=True)
        else:
            print("Callees (confirmed):")
            show(graph, graph.callees_of(node_id), use_source=False)
            print("Calls that could not be resolved (uncertain):")
            show(graph, graph.uncertain_callees_of(node_id), use_source=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.folder)
    except RepoLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    analysis = analyze_repository(root)
    try:
        graph = build_graph(analysis)
    finally:
        analysis.close()
    print_summary(graph, root)
    status = 0
    if args.limitations:
        print("\nWhat static analysis cannot resolve:")
        for line in LIMITATIONS:
            print(f"  - {line}")
    if args.edges:
        print("\nAll edges:")
        for r in graph.relations():
            arrow = f"{r.source_label} -> {r.target_label}"
            print(f"  {r.kind}/{r.certainty}  {arrow}  at {r.location}")
    if args.uncertain:
        print("\nUncertain relationships:")
        uncertain = [r for r in graph.relations() if r.certainty == "uncertain"]
        show(graph, uncertain, use_source=False)
    if args.callers:
        status = max(status, lookup(graph, args.callers, callers=True))
    if args.callees:
        status = max(status, lookup(graph, args.callees, callers=False))
    return status


if __name__ == "__main__":
    sys.exit(main())