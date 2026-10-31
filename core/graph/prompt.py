"""Prompt for graph-expanded answers: the Phase 3 prompt plus call-graph rules and notes."""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.chat.models import Evidence, RepoInfo
from core.chat.prompt import SYSTEM_PROMPT, USER_TEMPLATE, Prompt, format_evidence
from core.graph.expand import ExpansionResult, GraphEvidence

GRAPH_RULES = """\
CALL GRAPH CONTEXT
Some excerpts were added because a static call graph links them to a retrieved excerpt.
Their header says "via call graph" and names the call and the line where it appears.
- Those links are confirmed by reading the code text. They show that code CAN call other
  code. They do not prove that the call happens at runtime.
- Lines under CALL GRAPH NOTES are not excerpts. Never cite them and never state an
  UNCERTAIN relationship as a fact. If the question depends on one, say so under UNKNOWNS.
- When you describe a call, cite the excerpt that contains the call line.
"""


def graph_system_prompt() -> str:
    """The Phase 3 rules, with the call-graph rules inserted before the output format."""
    head, marker, tail = SYSTEM_PROMPT.partition("\nOUTPUT FORMAT")
    if not marker:
        return SYSTEM_PROMPT + "\n" + GRAPH_RULES
    return head + "\n" + GRAPH_RULES + marker + tail


def _defuse(code: str) -> str:
    """Stop repository code from faking our '--- ...' delimiter lines."""
    return re.sub(r"(?m)^---", "- - -", code)


def _graph_block(item: GraphEvidence) -> str:
    sym = item.symbol
    where = f"{sym.location} | {sym.kind} | {sym.qualname}"
    header = f"[{item.id}] {where} | via call graph, {item.via}"
    return f"{header}\n--- {item.id} CODE ---\n{_defuse(item.code)}\n--- END {item.id} ---"


def format_graph_evidence(evidence: Sequence[Evidence]) -> str:
    blocks = [
        _graph_block(e) if isinstance(e, GraphEvidence) else format_evidence([e])
        for e in evidence
    ]
    return "\n\n".join(blocks)


def build_graph_prompt(
    question: str, evidence: Sequence[Evidence], repo: RepoInfo, expansion: ExpansionResult
) -> Prompt:
    block = format_graph_evidence(evidence)
    if expansion.notes:
        lines = "\n".join(f"- {' '.join(note.split())}" for note in expansion.notes)
        block += "\n\nCALL GRAPH NOTES (not excerpts; do not cite)\n" + lines
    user = USER_TEMPLATE.format(
        repo_name=repo.name,
        symbol_count=repo.symbol_count,
        excerpt_count=len(evidence),
        evidence_block=block,
        question=question,
    )
    return Prompt(system=graph_system_prompt(), user=user)