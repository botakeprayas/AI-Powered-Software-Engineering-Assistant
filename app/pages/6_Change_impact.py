"""Streamlit page (Phase 7): change-impact analysis for a pasted diff.

Ranks are HEURISTIC: they order changed code for review and are not probabilities.
Known dependents come from confirmed graph edges, inferred ones from uncertain edges.
"""

import dataclasses
import difflib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.graph.expand import load_graph_context  # noqa: E402
from core.impact.analyze import (  # noqa: E402
    ImpactReport,
    UnitImpact,
    analyze_impact,
    compare_files,
    compare_tests,
    default_signals,
    load_impact_config,
)
from core.impact.changes import IndexView, map_diff  # noqa: E402
from core.impact.diff_parser import parse_diff  # noqa: E402
from core.impact.render import (  # noqa: E402
    dependent_rows,
    factor_rows,
    file_rows,
    recommended_rows,
    skipped_rows,
    unit_location,
    unit_rows,
)
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.settings import load_settings  # noqa: E402

st.set_page_config(page_title="Change impact", page_icon="🌊", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("impact_page")

BASE_CHOICES = {
    "auto": "Detect: the version before the change, else the version that already has it",
    "old": "The folder holds the version BEFORE the change (a proposed change)",
    "new": "The folder already contains the change",
}


def load_repository(folder_text: str) -> None:
    try:
        root = validate_folder(folder_text)
        with st.spinner("Indexing symbols and building the call graph..."):
            context = load_graph_context(root)
            view = IndexView.from_context(context)
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except Exception:
        logger.exception("Loading the repository failed")
        st.sidebar.error("Loading failed unexpectedly. Details are in the terminal log.")
    else:
        st.session_state.update(impact_context=context, impact_view=view, impact_report=None)


def fill_sample() -> None:
    path = settings.data_dir / "impact_sample" / "pricing" / "rates.py"
    try:
        old = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        st.session_state["impact_diff"] = ""
        return
    new = list(old)
    if len(new) >= 8:
        new[7] = "    return round(amount * (1 + TAX_RATE), 3)\n"
    body = "".join(difflib.unified_diff(old, new, "a/pricing/rates.py", "b/pricing/rates.py"))
    st.session_state["impact_diff"] = "diff --git a/pricing/rates.py b/pricing/rates.py\n" + body


def table_or_none(rows: list[dict]) -> None:
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("none")


def show_unit(impact: UnitImpact) -> None:
    unit = impact.unit
    title = (
        f"{impact.rank}. {unit.qualname} · {impact.band} "
        f"(heuristic rank {impact.score} of 100)"
    )
    with st.expander(title):
        st.caption(f"{unit_location(unit)} · {unit.describe()}")
        st.dataframe(factor_rows(impact), hide_index=True)
        for note in impact.notes:
            st.caption(f"Note: {note}")
        for finding in impact.findings:
            st.caption(f"Ruff {finding.code} at line {finding.line}: {finding.message}")
        st.markdown("**Known dependents (confirmed edges)**")
        table_or_none(dependent_rows(impact.known))
        st.markdown("**Inferred dependents (an uncertain link, NOT confirmed)**")
        table_or_none(dependent_rows(impact.inferred))


def show_comparison(report: ImpactReport) -> None:
    st.subheader("Compare with your hand-checked list (optional)")
    st.caption("Type what you worked out by hand, one per line, and see how the analysis did.")
    left, right = st.columns(2)
    files_text = left.text_area("Affected files", key="impact_expect_files", height=110)
    tests_text = right.text_area("Tests to run", key="impact_expect_tests", height=110)
    pairs = (
        ("Affected files", files_text, compare_files),
        ("Tests to run", tests_text, compare_tests),
    )
    for title, text, compare in pairs:
        wanted = [line.strip().replace("\\", "/") for line in text.splitlines() if line.strip()]
        if not wanted:
            continue
        result = compare(report, wanted)
        st.markdown(f"**{title}:** {result.summary()}")
        if result.missed:
            st.warning("Missed: " + ", ".join(result.missed))
        if result.found_inferred_only:
            st.info("Found only as inferred: " + ", ".join(result.found_inferred_only))


def show_report(report: ImpactReport) -> None:
    st.warning(report.disclaimer)
    for note in report.notes:
        st.caption(f"Note: {note}")
    if not report.units:
        st.info("No changed Python code could be mapped, so no impact was computed.")
    else:
        st.subheader("Changed code, ranked")
        st.dataframe(unit_rows(report), hide_index=True)
        for impact in report.units:
            show_unit(impact)
    st.subheader("Known dependencies (confirmed edges)")
    st.caption("Every link in these chains is a confirmed edge in the dependency graph.")
    table_or_none(file_rows(report.affected_known))
    st.subheader("Inferred impacts (not confirmed)")
    st.caption("Each of these chains includes at least one uncertain link. Treat them as leads.")
    table_or_none(file_rows(report.affected_inferred))
    st.subheader("Recommended tests")
    table_or_none(recommended_rows(report.tests))
    if report.test_command:
        st.code(report.test_command, language="powershell")
    if report.skipped:
        st.subheader("Not analyzed")
        st.dataframe(skipped_rows(report), hide_index=True)
    show_comparison(report)
    with st.expander("What this analysis cannot see"):
        for line in report.limitations:
            st.markdown(f"- {line}")


st.title("Change impact")
st.caption(
    "Phase 7: paste a diff and see what it may affect. Ranks are heuristic, not probabilities."
)

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.data_dir / "impact_sample"))
    if st.button("Load repository", type="primary"):
        load_repository(folder_text)
    st.header("Signals")
    use_git = st.checkbox("Use Git history (churn)", value=True, key="impact_git")
    use_ruff = st.checkbox("Run Ruff (static analysis)", value=True, key="impact_ruff")
    depth = st.slider("Trace depth", 1, 5, 4, key="impact_depth")

try:
    config = load_impact_config()
    traversal = dataclasses.replace(config.traversal, max_depth=depth)
    config = dataclasses.replace(
        config,
        traversal=traversal,
        use_git=config.use_git and use_git,
        use_ruff=config.use_ruff and use_ruff,
    )
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

context = st.session_state.get("impact_context")
if context is None:
    st.info(
        "Choose a folder in the sidebar and press **Load repository**. "
        "The sample is `data/impact_sample`."
    )
    st.stop()

st.caption(f"Repository: {context.root} · {len(context.symbols)} symbols indexed")
st.text_area("Pasted diff (output of git diff)", height=240, key="impact_diff")
base = st.selectbox(
    "The indexed folder is...", list(BASE_CHOICES), format_func=BASE_CHOICES.get, key="impact_base"
)
left, right = st.columns([1, 5])
left.button("Use sample", on_click=fill_sample)
if right.button("Analyze impact", type="primary"):
    with st.spinner("Mapping the diff and tracing the dependency graph..."):
        try:
            parsed = parse_diff(st.session_state.get("impact_diff", ""))
            changes = map_diff(parsed, st.session_state["impact_view"], base)
            signals = default_signals(context.root, config)
            result = analyze_impact(
                changes, context, st.session_state["impact_view"], config, signals
            )
        except Exception:
            logger.exception("Impact analysis failed")
            st.error("The analysis failed unexpectedly. Details are in the terminal log.")
        else:
            st.session_state["impact_report"] = result

report: ImpactReport | None = st.session_state.get("impact_report")
if report is not None:
    show_report(report)