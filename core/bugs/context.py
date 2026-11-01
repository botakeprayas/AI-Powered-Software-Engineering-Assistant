"""Retrieval for a bug: the failing code, its graph neighbours and related tests.

Every excerpt records why it was included. Frames that cannot be mapped to indexed code
are listed with a reason instead of being skipped silently.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.bugs.mapping import MAPPED, MappedFrame, is_test_path, pick_primary, select_frames
from core.bugs.traceback_parser import ParsedTraceback
from core.chat.models import Evidence, QAConfig
from core.graph.expand import ExpansionConfig, ExpansionResult, GraphContext, expand_evidence
from core.semantic.models import Symbol

logger = logging.getLogger(__name__)

TEST_KINDS = ("function", "method", "async_function", "async_method")


@dataclass(frozen=True)
class BugContextConfig:
    max_frames: int = 3  # repository frames used as evidence
    max_tests: int = 2  # related tests added by name matching
    max_total_chars: int = 9000  # all excerpts together


@dataclass(frozen=True)
class BugEvidence(Evidence):
    """An excerpt chosen because of the traceback, with the reason it was chosen."""

    role: str = ""  # failing frame, calling frame, test frame or related test
    why: str = ""
    plain_code: str = ""  # the same excerpt without the traceback marker (used for fixes)


@dataclass
class BugContext:
    evidence: list[Evidence] = field(default_factory=list)
    primary: MappedFrame | None = None
    expansion: ExpansionResult = field(default_factory=ExpansionResult)
    refused: list[MappedFrame] = field(default_factory=list)  # frames not mapped, with reasons
    notes: list[str] = field(default_factory=list)

    def related_test_ids(self) -> list[str]:
        """pytest node ids of every test function in the evidence."""
        ids: list[str] = []
        for item in self.evidence:
            sym = item.symbol
            is_test = is_test_path(sym.file) and sym.kind in TEST_KINDS
            if is_test and sym.name.startswith("test"):
                node = f"{sym.file}::{sym.qualname.replace('.', '::')}"
                if node not in ids:
                    ids.append(node)
        return ids

    def editable_files(self) -> list[str]:
        """Source files shown as evidence. A fix may only edit these (never test files)."""
        return sorted({e.symbol.file for e in self.evidence if not is_test_path(e.symbol.file)})


def one_line(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def window_code(
    symbol: Symbol, line: int, limit: int, mark: str = ""
) -> tuple[str, str, bool]:
    """Return (marked code, plain code, was_cut). The traceback line gets `mark` appended.

    Long code is cut to a window around the failing line, with omitted-lines markers.
    """
    lines = symbol.code.split("\n")
    focus = min(max(line - symbol.start_line, 0), len(lines) - 1)
    lo, hi = 0, len(lines) - 1
    if len(symbol.code) > limit:
        lo = hi = focus
        used = len(lines[focus])
        while True:
            grew = False
            if lo > 0 and used + len(lines[lo - 1]) + 1 <= limit:
                lo -= 1
                used += len(lines[lo]) + 1
                grew = True
            if hi < len(lines) - 1 and used + len(lines[hi + 1]) + 1 <= limit:
                hi += 1
                used += len(lines[hi]) + 1
                grew = True
            if not grew:
                break
    cut = lo > 0 or hi < len(lines) - 1

    def build(marked: bool) -> str:
        parts = []
        if lo > 0:
            parts.append(f"# [... {lo} line(s) omitted ...]")
        for index in range(lo, hi + 1):
            text = lines[index]
            if marked and mark and index == focus:
                text = f"{text}  # <-- {mark}"
            parts.append(text)
        if hi < len(lines) - 1:
            parts.append(f"# [... {len(lines) - 1 - hi} line(s) omitted ...]")
        return "\n".join(parts)

    return build(True), build(False), cut


def find_related_tests(
    context: GraphContext, targets: list[Symbol], exclude: set[str], limit: int
) -> list[tuple[Symbol, list[str]]]:
    """Tests that exercise the failing code, best first, each with the reasons it was picked."""
    candidates = [
        s for s in context.symbols.values()
        if is_test_path(s.file) and s.kind in TEST_KINDS and s.name.startswith("test")
        and s.key not in exclude
    ]
    by_key = {s.key: s for s in candidates}
    scores: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}

    def add(symbol: Symbol, points: int, reason: str) -> None:
        scores[symbol.key] = scores.get(symbol.key, 0) + points
        found = reasons.setdefault(symbol.key, [])
        if reason not in found:
            found.append(reason)

    for target in targets:
        try:
            callers = context.graph.callers_of(target.key)
        except Exception:  # noqa: BLE001 - name matching below still works
            callers = []
        for rel in callers:
            symbol = by_key.get(rel.source)
            if symbol is not None:
                add(symbol, 3, f"calls {target.qualname} at {rel.location} (confirmed)")
        stem = Path(target.file).stem
        mention = re.compile(rf"\b{re.escape(target.name)}\b")
        for symbol in candidates:
            if Path(symbol.file).stem in (f"test_{stem}", f"{stem}_test"):
                add(symbol, 2, f"is in {symbol.file}, which is named after {target.file}")
            if mention.search(symbol.code):
                add(symbol, 1, f"mentions {target.name}")
    ranked = sorted(
        (by_key[key] for key in scores),
        key=lambda s: (-scores[s.key], s.file, s.start_line),
    )
    return [(symbol, reasons[symbol.key]) for symbol in ranked[:limit]]


def build_bug_context(
    parsed: ParsedTraceback,
    mapped: list[MappedFrame],
    context: GraphContext,
    qa_config: QAConfig,
    expansion_config: ExpansionConfig | None = None,
    config: BugContextConfig | None = None,
) -> BugContext:
    """Failing frames first, then confirmed graph neighbours, then related tests."""
    config = config or BugContextConfig()
    expansion_config = expansion_config or ExpansionConfig()
    result = BugContext(primary=pick_primary(mapped))
    result.refused = [m for m in mapped if m.status != MAPPED]
    chosen = select_frames(mapped, limit=config.max_frames, include_test=True)
    if not chosen:
        result.notes.append(
            "No traceback frame could be mapped to indexed code, so there is no evidence."
        )
        return result

    last = parsed.frames[-1] if parsed.frames else None
    summary = one_line(f"{parsed.exception_type}: {parsed.exception_message}".strip(": "), 80)
    failing_mark = f"failing line: {summary or 'error'}"
    seeds: list[Evidence] = []
    repo_frames = 0
    for item in chosen:
        symbol = item.symbol
        if symbol is None:
            continue
        mark = failing_mark if item.frame == last else "traceback frame"
        marked, plain, cut = window_code(symbol, item.frame.line, qa_config.max_code_chars, mark)
        if item.is_test:
            role = "test frame"
        elif item is result.primary:
            role = "failing frame"
        else:
            role = "calling frame"
        repo_frames += 0 if item.is_test else 1
        seeds.append(
            BugEvidence(
                id=f"E{len(seeds) + 1}", symbol=symbol, code=marked, code_truncated=cut,
                semantic_score=None, keyword_rank=None, keyword_score=None,
                role=role, why=f"traceback frame at {item.location}", plain_code=plain,
            )
        )

    graph_config = dataclasses.replace(expansion_config, max_seeds=max(1, repo_frames))
    try:
        result.expansion = expand_evidence(context, seeds, qa_config, graph_config)
    except Exception:
        logger.exception("Call-graph expansion failed")
        result.notes.append("Call-graph expansion failed, so only traceback frames are used.")
    evidence: list[Evidence] = [*seeds, *result.expansion.evidence]

    targets = [e.symbol for e in seeds if not is_test_path(e.symbol.file)]
    have = {e.symbol.key for e in evidence}
    used = sum(len(e.code) for e in evidence)
    for symbol, reasons in find_related_tests(context, targets, have, config.max_tests):
        code, plain, cut = window_code(symbol, symbol.start_line, qa_config.max_code_chars)
        if used + len(code) > config.max_total_chars:
            result.notes.append(f"{symbol.qualname} not added: prompt size limit reached")
            continue
        used += len(code)
        evidence.append(
            BugEvidence(
                id=f"E{len(evidence) + 1}", symbol=symbol, code=code, code_truncated=cut,
                semantic_score=None, keyword_rank=None, keyword_score=None,
                role="related test", why="; ".join(reasons), plain_code=plain,
            )
        )
    result.evidence = evidence
    return result