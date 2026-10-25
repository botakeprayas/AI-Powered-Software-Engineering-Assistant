"""NetworkX dependency graph. Every edge carries kind, certainty, file, line and a reason."""

from __future__ import annotations

import networkx as nx

from core.graph.models import (
    CALLS,
    CONFIRMED,
    NODE_CLASS,
    NODE_FUNCTION,
    UNCERTAIN,
    GraphStats,
    Relation,
)


class DependencyGraph:
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()  # two nodes may be linked several times (different lines)
        self._seen: set[tuple[str, str, str, str, int]] = set()
        self.stats = GraphStats()

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        return self._g

    def has_node(self, node_id: str) -> bool:
        return node_id in self._g

    def node(self, node_id: str) -> dict:
        return dict(self._g.nodes[node_id])

    def label(self, node_id: str) -> str:
        if node_id not in self._g:
            return node_id
        return str(self._g.nodes[node_id].get("label", node_id))

    def add_node(self, node_id: str, kind: str, label: str, **attrs) -> None:
        if node_id in self._g:
            return
        self._g.add_node(node_id, kind=kind, label=label, **attrs)
        self.stats.nodes[kind] = self.stats.nodes.get(kind, 0) + 1

    def add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        certainty: str,
        file: str,
        line: int,
        reason: str,
        candidates: tuple[str, ...] = (),
    ) -> bool:
        """Add one located edge. Returns False for an exact duplicate."""
        if source not in self._g or target not in self._g:
            raise KeyError(f"edge {source!r} -> {target!r} refers to a missing node")
        signature = (source, target, kind, file, line)
        if signature in self._seen:
            return False
        self._seen.add(signature)
        self._g.add_edge(
            source, target, kind=kind, certainty=certainty, file=file, line=line,
            reason=reason, candidates=tuple(candidates),
        )
        key = f"{kind}/{certainty}"
        self.stats.edges[key] = self.stats.edges.get(key, 0) + 1
        return True

    def _relation(self, source: str, target: str, data: dict) -> Relation:
        return Relation(
            kind=data["kind"],
            certainty=data["certainty"],
            source=source,
            target=target,
            source_label=self.label(source),
            target_label=self.label(target),
            file=data["file"],
            line=data["line"],
            reason=data["reason"],
            candidates=tuple(data["candidates"]),
        )

    @staticmethod
    def _order(relation: Relation) -> tuple:
        return (relation.file, relation.line, relation.source, relation.target)

    def relations(self, kind: str | None = None, certainty: str | None = None) -> list[Relation]:
        found = []
        for source, target, data in self._g.edges(data=True):
            if kind is not None and data["kind"] != kind:
                continue
            if certainty is not None and data["certainty"] != certainty:
                continue
            found.append(self._relation(source, target, data))
        return sorted(found, key=self._order)

    def find_symbols(self, name: str) -> list[str]:
        """Node ids of functions/classes whose id or name matches ('checkout' or 'A.checkout')."""
        found = []
        for node_id, data in self._g.nodes(data=True):
            if data.get("kind") not in (NODE_FUNCTION, NODE_CLASS):
                continue
            label = str(data.get("label", ""))
            if name in (node_id, label) or label.endswith("." + name):
                found.append(node_id)
        return sorted(found)

    def callers_of(self, node_id: str) -> list[Relation]:
        """Confirmed callers only. A caller may be a file node (module-level code)."""
        if node_id not in self._g:
            return []
        found = [
            self._relation(u, v, d)
            for u, v, d in self._g.in_edges(node_id, data=True)
            if d["kind"] == CALLS and d["certainty"] == CONFIRMED
        ]
        return sorted(found, key=self._order)

    def callees_of(self, node_id: str) -> list[Relation]:
        """Confirmed callees only."""
        if node_id not in self._g:
            return []
        found = [
            self._relation(u, v, d)
            for u, v, d in self._g.out_edges(node_id, data=True)
            if d["kind"] == CALLS and d["certainty"] == CONFIRMED
        ]
        return sorted(found, key=self._order)

    def uncertain_callees_of(self, node_id: str) -> list[Relation]:
        """Calls made from this node that could not be resolved (target is an unresolved node)."""
        if node_id not in self._g:
            return []
        found = [
            self._relation(u, v, d)
            for u, v, d in self._g.out_edges(node_id, data=True)
            if d["kind"] == CALLS and d["certainty"] == UNCERTAIN
        ]
        return sorted(found, key=self._order)

    def possible_callers_of(self, node_id: str) -> list[Relation]:
        """Uncertain calls that MIGHT reach this node (it is only a candidate, not confirmed)."""
        return [r for r in self.relations(CALLS, UNCERTAIN) if node_id in r.candidates]