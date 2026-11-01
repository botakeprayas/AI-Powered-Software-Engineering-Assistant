"""Bug investigation workflow (Phase 6, part 2).

traceback -> evidence -> validated hypothesis -> proposed fix and regression-test draft.
Nothing here runs generated code. The fix is PROPOSED, NOT VERIFIED and the test is a
DRAFT, NOT RUN until the sandbox (part 3) has run them.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.bugs.context import BugContext, BugContextConfig, build_bug_context, one_line
from core.bugs.mapping import FrameMapper, MappedFrame
from core.bugs.prompt import (
    build_fix_prompt,
    build_hypothesis_prompt,
    build_test_prompt,
    build_test_retry_prompt,
    entry_call_text,
)
from core.bugs.proposal import (
    DRAFT_READY,
    FixProposal,
    RegressionDraft,
    build_fix_proposal,
    build_regression_draft,
)
from core.bugs.traceback_parser import ParsedTraceback, parse_traceback
from core.chat.models import Answer, Claim, QAConfig
from core.chat.parsing import build_answer
from core.graph.expand import ExpansionConfig, GraphContext
from core.graph.models import NODE_FILE
from core.repo.extractor import normalize_newlines
from core.validation.config import ValidationConfig
from core.validation.models import (
    LEVEL_LIMITED,
    LEVEL_SUFFICIENT,
    SUPPORTED,
    WEAK,
    ValidationReport,
)
from core.validation.report import validate_answer
from core.validation.snapshot import IndexSnapshot
from services.llm_service import LLMError, LLMService

logger = logging.getLogger(__name__)

RETRIEVAL_NOTE = (
    "Evidence was chosen from the traceback and the call graph, not by retrieval score, "
    "so the retrieval-score rule is switched off for this check."
)


@dataclass
class Investigation:
    traceback: ParsedTraceback
    mapped: list[MappedFrame]
    context: BugContext
    hypothesis: Answer | None = None
    report: ValidationReport | None = None
    fix: FixProposal | None = None
    test_draft: RegressionDraft | None = None
    steps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    is_mock: bool = False


def _indexed_files(context: GraphContext) -> set[str]:
    files = {s.file for s in context.symbols.values()}
    try:
        for node_id, data in context.graph.nx_graph.nodes(data=True):
            if data.get("kind") == NODE_FILE:
                files.add(node_id)
    except Exception:  # noqa: BLE001 - the files that have symbols are enough
        pass
    return files


def _claim_line(claim: Claim) -> str:
    return f"- {claim.text} [{', '.join(claim.evidence_ids)}]"


def hypothesis_block(answer: Answer, report: ValidationReport | None) -> tuple[str, int]:
    """Validated claims as prompt text, and how many hypothesis claims survived validation."""
    ok = None
    if report is not None:
        ok = {c.label for c in report.claims if c.final in (SUPPORTED, WEAK)}
    facts = [
        _claim_line(claim)
        for number, claim in enumerate(answer.facts, start=1)
        if ok is None or f"F{number}" in ok
    ]
    inferences: list[str] = []
    hypotheses = 0
    for number, claim in enumerate(answer.inferences, start=1):
        if ok is not None and f"I{number}" not in ok:
            continue
        inferences.append(_claim_line(claim))
        if claim.text.lower().startswith("hypothesis"):
            hypotheses += 1
    survived = hypotheses or len(inferences)
    text = "FACTS:\n" + ("\n".join(facts) or "- none")
    text += "\nHYPOTHESIS AND INFERENCES:\n" + ("\n".join(inferences) or "- none")
    return text, survived


def _reader(root: Path, allowed: set[str]) -> Callable[[str], str | None]:
    base = Path(root).resolve()

    def read(rel: str) -> str | None:
        if rel not in allowed:
            return None
        target = (base / rel).resolve()
        if base not in target.parents:
            return None
        try:
            return normalize_newlines(target.read_bytes().decode("utf-8-sig"))
        except (OSError, UnicodeDecodeError):
            return None

    return read


def _slug(parsed: ParsedTraceback, bug: BugContext) -> str:
    name = "bug"
    if bug.primary is not None and bug.primary.symbol is not None:
        name = bug.primary.symbol.name
    raw = f"{parsed.exception_type}_{name}".lower()
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")[:40] or "bug"


def _hypothesis(
    inv: Investigation, llm: LLMService, qa_config: QAConfig, repo_name: str,
    symbol_count: int, traceback_text: str,
) -> Answer | None:
    prompt = build_hypothesis_prompt(
        traceback_text, inv.traceback, inv.mapped, inv.context, repo_name, symbol_count
    )
    try:
        response = llm.generate(
            prompt.user, system=prompt.system, temperature=qa_config.temperature,
            max_tokens=qa_config.max_answer_tokens,
        )
    except LLMError as exc:
        inv.errors.append(str(exc))
        return None
    parsed = inv.traceback
    question = one_line(f"Root cause of {parsed.exception_type}: {parsed.exception_message}", 200)
    return build_answer(question, response, inv.context.evidence)


def _validate(
    inv: Investigation, answer: Answer, context: GraphContext,
    config: ValidationConfig, judge: LLMService,
) -> ValidationReport | None:
    snapshot = IndexSnapshot(
        context.root, {s.key: (s.file, s.start_line, s.end_line) for s in context.symbols.values()}
    )
    relaxed = dataclasses.replace(config, min_strong_evidence=0)
    try:
        report = validate_answer(answer, snapshot, relaxed, judge)
    except Exception:
        logger.exception("Hypothesis validation failed")
        inv.errors.append("Validation could not run, so the hypothesis is NOT validated.")
        return None
    report.notes.append(RETRIEVAL_NOTE)
    return report


def _fix_step(
    inv: Investigation, block: str, context: GraphContext, llm: LLMService
) -> None:
    editable = inv.context.editable_files()
    shown = [e for e in inv.context.evidence if e.symbol.file in set(editable)]
    prompt = build_fix_prompt(block, shown, editable)
    try:
        response = llm.generate(prompt.user, system=prompt.system, temperature=0.0, max_tokens=700)
    except LLMError as exc:
        inv.errors.append(f"Fix step: {exc}")
        return
    if response.truncated:
        inv.steps.append("The fix answer hit the model's length limit.")
    inv.fix = build_fix_proposal(response.text, _reader(context.root, set(editable)), editable)


def _ask_test(
    inv: Investigation, llm: LLMService, prompt, context: GraphContext, slug: str
) -> RegressionDraft | None:
    """One model call for the test draft. Returns None (and records the error) on failure."""
    try:
        response = llm.generate(prompt.user, system=prompt.system, temperature=0.0, max_tokens=700)
    except LLMError as exc:
        inv.errors.append(f"Test step: {exc}")
        return None
    root = Path(context.root)
    return build_regression_draft(response.text, slug, lambda rel: (root / rel).exists())


def _why_unusable(draft: RegressionDraft) -> str:
    if draft.problems:
        return "It was rejected: " + " ".join(draft.problems)
    return "It declined: " + (draft.reason or "no reason was given")


def _test_step(
    inv: Investigation, block: str, context: GraphContext, llm: LLMService
) -> None:
    """Ask for a test draft. If it is declined or rejected, ask once more with the reason."""
    diff = inv.fix.diff if inv.fix is not None and inv.fix.usable else ""
    slug = _slug(inv.traceback, inv.context)
    entry = entry_call_text(inv.mapped)
    prompt = build_test_prompt(block, inv.context.evidence, diff, slug, entry)
    draft = _ask_test(inv, llm, prompt, context, slug)
    if draft is None:
        return
    inv.test_draft = draft
    if draft.status == DRAFT_READY:
        return
    why = _why_unusable(draft)
    note = f"The first test draft was not usable ({why}); the model was asked once more."
    inv.steps.append(note)
    second = _ask_test(inv, llm, build_test_retry_prompt(prompt, why), context, slug)
    if second is not None:
        inv.test_draft = second

def _propose(
    inv: Investigation, answer: Answer, context: GraphContext, llm: LLMService
) -> None:
    if llm.is_mock:
        inv.steps.append(
            "The MOCK provider does not write fixes or tests, so those steps were skipped."
        )
        return
    report = inv.report
    if report is None or report.policy is None:
        inv.steps.append("No fix was proposed because validation did not run.")
        return
    if report.policy.level not in (LEVEL_SUFFICIENT, LEVEL_LIMITED):
        inv.steps.append(f"No fix was proposed: the evidence level is {report.policy.level}.")
        return
    block, survived = hypothesis_block(answer, report)
    if survived == 0:
        inv.steps.append("No fix was proposed: no root-cause hypothesis passed validation.")
        return
    _fix_step(inv, block, context, llm)
    _test_step(inv, block, context, llm)


def investigate(
    traceback_text: str,
    context: GraphContext,
    llm: LLMService,
    *,
    qa_config: QAConfig | None = None,
    validation_config: ValidationConfig | None = None,
    expansion_config: ExpansionConfig | None = None,
    bug_config: BugContextConfig | None = None,
    judge: LLMService | None = None,
    repo_name: str | None = None,
) -> Investigation:
    """Investigate a failure. Expected problems become messages on the result, not exceptions."""
    qa_config = qa_config or QAConfig()
    validation_config = validation_config or ValidationConfig()
    parsed = parse_traceback(traceback_text)
    inv = Investigation(
        parsed, [], BugContext(), llm_provider=llm.provider, llm_model=llm.model_name,
        is_mock=llm.is_mock,
    )
    inv.steps.extend(f"Traceback: {warning}" for warning in parsed.warnings)
    if not parsed.has_frames:
        inv.errors.append(
            "No 'File ..., line N' references were found, so there is nothing to investigate."
        )
        return inv
    symbols = list(context.symbols.values())
    mapper = FrameMapper(symbols, _indexed_files(context), root_hint=str(context.root))
    inv.mapped = mapper.map(parsed.frames)
    inv.context = build_bug_context(
        parsed, inv.mapped, context, qa_config, expansion_config, bug_config
    )
    inv.steps.extend(inv.context.notes)
    if not inv.context.evidence:
        inv.errors.append(
            "No traceback frame maps to indexed code (see the frame reasons), "
            "so there is no evidence to investigate."
        )
        return inv
    name = repo_name or Path(context.root).name
    answer = _hypothesis(inv, llm, qa_config, name, len(symbols), traceback_text)
    if answer is None:
        return inv
    inv.hypothesis = answer
    inv.report = _validate(
        inv, answer, context, validation_config, judge if judge is not None else llm
    )
    _propose(inv, answer, context, llm)
    return inv
