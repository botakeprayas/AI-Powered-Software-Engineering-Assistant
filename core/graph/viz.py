"""Dependency visualization with matplotlib: offline, static, no JavaScript.

Files are columns and functions/classes are boxes in them. Solid arrows are confirmed calls,
dashed arrows are uncertain calls that end in an 'unresolved' box (never a guessed target).
"""

from __future__ import annotations

import io
from collections import Counter, deque
from dataclasses import dataclass, field

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from core.graph.graph import DependencyGraph
from core.graph.models import (
    CALLS,
    CONFIRMED,
    NODE_CLASS,
    NODE_FILE,
    NODE_FUNCTION,
    NODE_UNRESOLVED,
    UNCERTAIN,
    Relation,
)

NODE_COLORS = {
    NODE_FUNCTION: "#cfe5f7",
    NODE_CLASS: "#e2dcf7",
    NODE_FILE: "#e6e6e6",
    NODE_UNRESOLVED: "#fde2cf",
}
CONFIRMED_COLOR = "#2b5f8a"
UNCERTAIN_COLOR = "#c2571a"
COLUMN_GAP = 3.4
ROW_GAP = 1.1
UNRESOLVED_COLUMN = "unresolved (not guessed)"
EXTERNAL_COLUMN = "external"


@dataclass
class GraphView:
    nodes: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    truncated: bool = False  # more nodes existed than max_nodes


def _bfs(adjacency: dict[str, set[str]], start: str, depth: int) -> list[str]:
    order = [start]
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        node, hop = frontier.popleft()
        if hop >= depth:
            continue
        for other in sorted(adjacency.get(node, ())):
            if other not in seen:
                seen.add(other)
                order.append(other)
                frontier.append((other, hop + 1))
    return order


def select_view(
    graph: DependencyGraph,
    focus: str | None = None,
    depth: int = 1,
    certainties: tuple[str, ...] = (CONFIRMED, UNCERTAIN),
    max_nodes: int = 40,
) -> GraphView:
    """Pick the call relationships to draw: the whole graph, or the area around `focus`."""
    if focus is not None and not graph.has_node(focus):
        raise KeyError(f"unknown node {focus!r}")
    relations = [r for r in graph.relations(CALLS) if r.certainty in certainties]
    adjacency: dict[str, set[str]] = {}
    degree: Counter[str] = Counter()
    for rel in relations:
        adjacency.setdefault(rel.source, set()).add(rel.target)
        adjacency.setdefault(rel.target, set()).add(rel.source)
        degree[rel.source] += 1
        degree[rel.target] += 1
    if focus is None:
        order = sorted(adjacency, key=lambda node: (-degree[node], node))
    else:
        order = _bfs(adjacency, focus, depth)
    keep = set(order[:max_nodes])
    shown = [r for r in relations if r.source in keep and r.target in keep]
    return GraphView(sorted(keep), shown, truncated=len(order) > max_nodes)


def _column(graph: DependencyGraph, node_id: str) -> str:
    data = graph.node(node_id)
    if data.get("kind") == NODE_UNRESOLVED:
        return UNRESOLVED_COLUMN
    return str(data.get("file") or EXTERNAL_COLUMN)


def _column_order(name: str) -> tuple[int, str]:
    if name == UNRESOLVED_COLUMN:
        return (2, name)
    if name == EXTERNAL_COLUMN:
        return (1, name)
    return (0, name)


def _row_order(graph: DependencyGraph, node_id: str) -> tuple[int, int, str]:
    data = graph.node(node_id)
    is_symbol = 0 if data.get("kind") == NODE_FILE else 1
    return (is_symbol, int(data.get("line") or 0), graph.label(node_id))


def _short(text: str, limit: int = 26) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def draw_view(graph: DependencyGraph, view: GraphView, title: str = "") -> Figure:
    """Draw a view. Uses the object API (no pyplot), so no window or backend is needed."""
    if not view.nodes:
        fig = Figure(figsize=(6, 2), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No call relationships to show", ha="center", va="center")
        return fig
    columns: dict[str, list[str]] = {}
    for node_id in view.nodes:
        columns.setdefault(_column(graph, node_id), []).append(node_id)
    names = sorted(columns, key=_column_order)
    pos: dict[str, tuple[float, float]] = {}
    max_rows = 1
    for col, name in enumerate(names):
        members = sorted(columns[name], key=lambda node: _row_order(graph, node))
        max_rows = max(max_rows, len(members))
        for row, node_id in enumerate(members):
            pos[node_id] = (col * COLUMN_GAP, -row * ROW_GAP)

    width = min(16.0, max(6.0, len(names) * 3.2 + 0.8))
    height = min(14.0, max(3.6, max_rows * 1.0 + 2.0))
    fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(-1.9, (len(names) - 1) * COLUMN_GAP + 1.9)
    ax.set_ylim(-(max_rows - 1) * ROW_GAP - 0.8, 1.4)
    for col, name in enumerate(names):
        ax.text(col * COLUMN_GAP, 0.8, _short(name, 30), ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#555555")

    drawn: set[tuple[str, str, str]] = set()
    for rel in view.relations:
        signature = (rel.source, rel.target, rel.certainty)
        if rel.source == rel.target or signature in drawn:
            continue
        drawn.add(signature)
        sure = rel.certainty == CONFIRMED
        ax.annotate(
            "",
            xy=pos[rel.target],
            xytext=pos[rel.source],
            arrowprops={
                "arrowstyle": "-|>",
                "color": CONFIRMED_COLOR if sure else UNCERTAIN_COLOR,
                "lw": 1.2,
                "linestyle": "-" if sure else "--",
                "shrinkA": 16,
                "shrinkB": 16,
                "connectionstyle": "arc3,rad=0.12",
            },
            zorder=1,
        )
    for node_id, (x, y) in pos.items():
        kind = graph.node(node_id).get("kind")
        label = "(module-level code)" if kind == NODE_FILE else _short(graph.label(node_id))
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": NODE_COLORS.get(kind, "#eeeeee"),
                      "edgecolor": "#555555", "linewidth": 0.8})
    handles = [
        Line2D([0], [0], color=CONFIRMED_COLOR, lw=1.5, label="confirmed call"),
        Line2D([0], [0], color=UNCERTAIN_COLOR, lw=1.5, ls="--",
               label="uncertain call (target unresolved)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.1)
    return fig


def figure_png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    return buffer.getvalue()