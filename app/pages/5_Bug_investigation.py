"""Streamlit page (Phase 6): investigate a failure from an error message or stack trace.

Fixes are PROPOSED, NOT VERIFIED and tests are DRAFTS until the sandbox has run them.
Running the draft test is a separate button, because it executes model-written code.
"""

import dataclasses
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.bugs.investigate import Investigation, investigate  # noqa: E402
from core.bugs.proposal import DRAFT_LABEL, PROPOSED_LABEL  # noqa: E402
from core.bugs.render import describe_evidence  # noqa: E402
from core.bugs.verify import (  # noqa: E402
    FIX_INEFFECTIVE,
    LIMITS_TEXT,
    REGRESSION,
    SANDBOX_TEXT,
    VERIFIED,
    VerificationReport,
    verify_investigation,
)
from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.graph.expand import load_expansion_config, load_graph_context  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.sandbox.config import load_sandbox_config  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402
from core.validation.models import (  # noqa: E402
    LEVEL_DISCLAIMER,
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    LEVEL_SUFFICIENT,
    UNSUPPORTED,
    WEAK,
)

st.set_page_config(page_title="Bug investigation", page_icon="🐞", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("bug_page")


def load_repository(folder_text: str) -> None:
    try:
        root = validate_folder(folder_text)
        with st.spinner("Indexing symbols and building the call graph..."):
            context = load_graph_context(root)
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except Exception:
        logger.exception("Loading the repository failed")
        st.sidebar.error("Loading failed unexpectedly. Details are in the terminal log.")
    else:
        st.session_state.update(bug_context=context, bug_root=root, bug_inv=None, bug_ver=None)


def fill_sample() -> None:
    path = settings.data_dir / "bug_sample" / "traceback.txt"
    try:
        st.session_state["bug_text"] = path.read_text(encoding="utf-8")
    except OSError:
        st.session_state["bug_text"] = ""


def show_frames(inv: Investigation) -> None:
    rows = [
        {
            "#": number,
            "location": item.location,
            "function": item.frame.func or "?",
            "status": item.status,
            "maps to": item.symbol.qualname if item.symbol else "-",
            "why": item.reason,
        }
        for number, item in enumerate(inv.mapped, start=1)
    ]
    st.dataframe(rows, hide_index=True)


def show_evidence(inv: Investigation) -> None:
    if not inv.context.evidence:
        st.caption("No evidence: no traceback frame mapped to indexed code.")
        return
    for item in inv.context.evidence:
        sym = item.symbol
        st.markdown(f"**{item.id}** · `{item.location}` · {sym.kind} `{sym.qualname}`")
        st.caption(describe_evidence(item))
        st.code(item.code, language="python")
    for note in inv.context.expansion.notes:
        st.caption(f"Call-graph note (not evidence): {note}")


def show_hypothesis(inv: Investigation) -> None:
    answer, report = inv.hypothesis, inv.report
    if answer is None:
        return
    st.subheader("Root-cause hypothesis")
    st.caption(
        "A hypothesis explains why the failure happens. "
        "It is not a fix, and it is not verified."
    )
    st.markdown(answer_to_markdown(answer))
    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")
    if report is None:
        st.warning("Validation did not run, so this hypothesis is NOT validated.")
        return
    policy = report.policy
    if policy is not None and policy.level != LEVEL_NOT_APPLICABLE:
        text = f"Evidence level: {policy.level} (rule-based)"
        if policy.level == LEVEL_SUFFICIENT:
            st.success(text)
        elif policy.level == LEVEL_LIMITED:
            st.warning(text)
        elif policy.level == LEVEL_INSUFFICIENT:
            st.error(text)
        st.caption(LEVEL_DISCLAIMER)
        for limitation in policy.limitations:
            st.warning(f"Limitation: {limitation}")
    for check in report.claims:
        if check.final in (UNSUPPORTED, WEAK):
            title = "Unsupported conclusion" if check.final == UNSUPPORTED else "Weakly supported"
            st.warning(f"{title} ({check.label}): {check.text}")
            for reason in check.reasons:
                st.caption(reason)


def show_fix(inv: Investigation, ver: VerificationReport | None) -> None:
    fix = inv.fix
    if fix is None:
        return
    verified = ver is not None and ver.verified
    st.subheader("Minimal fix")
    if verified:
        st.success("Fix status: VERIFIED BY THESE TESTS (read the limits under the results)")
    else:
        st.warning(f"Fix status: {PROPOSED_LABEL}")
    if fix.rationale:
        st.caption(f"Rationale: {fix.rationale}")
    if fix.diff:
        st.caption(
            f"{fix.changed_lines} changed line(s). "
            "The diff was computed by code, not written by the model."
        )
        st.code(fix.diff, language="diff")
    elif fix.status == "none":
        st.info(f"The model declined to propose a fix: {fix.rationale}")
    for problem in fix.problems:
        st.error(f"Rejected: {problem}")


def show_test_draft(inv: Investigation, ver: VerificationReport | None) -> None:
    draft = inv.test_draft
    if draft is None:
        return
    ran = ver is not None and ver.before is not None
    st.subheader("Regression test")
    status = "RAN IN THE SANDBOX (see the results)" if ran else DRAFT_LABEL
    st.warning(f"Test status: {status}")
    if draft.path:
        st.caption(
            f"It would be saved as `{draft.path}`, a new file. "
            "Nothing is written to your project."
        )
    if draft.reason:
        st.info(f"The model declined to write a test: {draft.reason}")
    if draft.code:
        st.code(draft.code, language="python")
    for problem in draft.problems:
        st.error(f"Rejected: {problem}")


def show_verification(ver: VerificationReport) -> None:
    st.subheader("Sandbox verification")
    if ver.verdict == VERIFIED:
        st.success(ver.headline)
    elif ver.verdict in (FIX_INEFFECTIVE, REGRESSION):
        st.error(ver.headline)
    else:
        st.warning(ver.headline)
    for reason in ver.reasons:
        st.caption(reason)
    for note in ver.notes:
        st.caption(note)
    for title, run in (("Before the fix", ver.before), ("After the fix", ver.after)):
        if run is None:
            continue
        summary = run.summary.text if run.summary else "no summary"
        st.markdown(
            f"**{title}**: {run.status} · exit code {run.returncode} · {summary} · "
            f"{run.duration:.1f}s"
        )
        if run.outcomes:
            rows = [{"test": t, "outcome": o} for t, o in sorted(run.outcomes.items())]
            st.dataframe(rows, hide_index=True)
        with st.expander(f"Real pytest output ({title.lower()})"):
            st.code(run.stdout or "(empty)", language=None)
            if run.stderr.strip():
                st.code(run.stderr, language=None)
    st.caption(LIMITS_TEXT)
    st.caption(SANDBOX_TEXT)


st.title("Bug investigation")
st.caption(
    "Phase 6: a hypothesis with evidence, a proposed fix and a test draft. "
    "Nothing is trusted until it is run."
)

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.data_dir / "bug_sample"))
    if st.button("Load repository", type="primary"):
        load_repository(folder_text)
    st.header("Language model")
    use_mock = st.checkbox("Use MOCK provider (no model needed)", value=False)
    use_judge = st.checkbox("Use LLM judge (one extra model call)", value=True)

try:
    llm = create_llm_from_env(provider_override="mock" if use_mock else None)
    qa_config = load_qa_config()
    validation_config = dataclasses.replace(load_validation_config(), judge_enabled=use_judge)
    expansion_config = load_expansion_config()
    sandbox_config = load_sandbox_config()
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

st.sidebar.caption(f"Provider: {llm.provider} · Model: {llm.model_name}")
if llm.is_mock:
    st.sidebar.warning("MOCK provider: it writes no fix or test, so nothing can be verified.")
elif not llm.is_local:
    st.sidebar.warning(f"Code excerpts are sent to {llm.endpoint}, not kept on this computer.")

context = st.session_state.get("bug_context")
if context is None:
    st.info(
        "Choose a folder in the sidebar and press **Load repository**. "
        "The sample is `data/bug_sample`."
    )
    st.stop()

st.caption(
    f"Repository: {st.session_state['bug_root']} · {len(context.symbols)} symbols indexed"
)
st.text_area("Error message or stack trace", height=220, key="bug_text")
left, right = st.columns([1, 5])
left.button("Use sample", on_click=fill_sample)
if right.button("Investigate", type="primary"):
    with st.spinner("Parsing the traceback, gathering evidence and asking the model..."):
        try:
            result = investigate(
                st.session_state.get("bug_text", ""), context, llm,
                qa_config=qa_config, validation_config=validation_config,
                expansion_config=expansion_config,
            )
        except Exception:
            logger.exception("Investigation failed")
            st.error("The investigation failed unexpectedly. Details are in the terminal log.")
        else:
            st.session_state.update(bug_inv=result, bug_ver=None)

inv: Investigation | None = st.session_state.get("bug_inv")
if inv is None:
    st.stop()
ver: VerificationReport | None = st.session_state.get("bug_ver")

for message in inv.errors:
    st.error(message)
for step in inv.steps:
    st.caption(step)
if inv.mapped:
    st.subheader("Traceback frames")
    show_frames(inv)
if inv.context.evidence:
    count = len(inv.context.evidence)
    with st.expander(f"Evidence ({count} excerpt(s), each with the reason it was chosen)"):
        show_evidence(inv)
show_hypothesis(inv)
show_fix(inv, ver)
show_test_draft(inv, ver)

if inv.test_draft is not None and inv.test_draft.usable:
    st.caption(
        "Running the test executes model-written code. It runs only in a temporary copy of "
        "the repository, never in your project folder."
    )
    if st.button("Run tests in sandbox"):
        with st.spinner("Running the draft test before and after the fix..."):
            try:
                outcome = verify_investigation(inv, st.session_state["bug_root"], sandbox_config)
            except Exception:
                logger.exception("Verification failed")
                st.error("Verification failed unexpectedly. Details are in the terminal log.")
            else:
                st.session_state["bug_ver"] = outcome
                ver = outcome
if ver is not None:
    show_verification(ver)