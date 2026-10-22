"""Streamlit UI (Phase 1): analyze a local folder or an uploaded zip.

No LLM, no embeddings and no API key are used in this phase.
Run only on your own machine: the folder-path box can read any folder you can read.
"""

import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root. Add the root so `import core` works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.logging_config import setup_logging  # noqa: E402
from core.repo.index import SEARCH_FIELDS  # noqa: E402
from core.repo.loader import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    RepoLoadError,
    load_zip_to_temp,
    validate_folder,
)
from core.repo.pipeline import RepoAnalysis, analyze_repository  # noqa: E402
from core.settings import load_settings  # noqa: E402

st.set_page_config(page_title="AI Software Engineering Assistant", page_icon="🛠️", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("app")


def release_previous() -> None:
    """Free the previous analysis (SQLite connection and any temp folder from a zip)."""
    previous = st.session_state.pop("analysis", None)
    if previous is not None:
        previous.close()
    temp_dir = st.session_state.pop("temp_dir", None)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


def store(analysis: RepoAnalysis, label: str, temp_dir: Path | None) -> None:
    release_previous()
    st.session_state["analysis"] = analysis
    st.session_state["label"] = label
    st.session_state["temp_dir"] = str(temp_dir) if temp_dir else None


def analyze_folder(folder_text: str) -> None:
    root = validate_folder(folder_text)
    store(analyze_repository(root), label=str(root), temp_dir=None)


def analyze_zip(upload) -> None:
    loaded = load_zip_to_temp(upload)
    try:
        analysis = analyze_repository(loaded.root, extra_skipped=loaded.skipped)
    except Exception:
        loaded.cleanup()
        raise
    store(analysis, label=upload.name, temp_dir=loaded.temp_dir)


def guarded(action, *args) -> None:
    """Run an action; show problems in the sidebar instead of crashing the page."""
    try:
        with st.spinner("Scanning repository..."):
            action(*args)
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except Exception:
        logger.exception("Unexpected error while analyzing the repository")
        st.sidebar.error("Unexpected error while analyzing. Details are in the terminal log.")


st.title("AI-Powered Software Engineering Assistant")
st.caption("Phase 1: scan a Python repository, extract symbols, search by keyword. No LLM needed.")

with st.sidebar:
    st.header("Repository")
    source = st.radio("Source", ["Local folder path", "Upload a .zip"])
    if source == "Local folder path":
        folder_text = st.text_input("Folder path", value=str(settings.sample_repo_dir))
        if st.button("Analyze", type="primary", key="analyze_folder"):
            guarded(analyze_folder, folder_text)
    else:
        upload = st.file_uploader("Zip of a small Python repository", type=["zip"])
        if upload is not None and st.button("Analyze", type="primary", key="analyze_zip"):
            if upload.size > MAX_UPLOAD_BYTES:
                st.error(f"Zip is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            else:
                guarded(analyze_zip, upload)

analysis: RepoAnalysis | None = st.session_state.get("analysis")
if analysis is None:
    st.info(
        "Choose a folder or upload a zip in the sidebar, then press **Analyze**. "
        "The default path is the built-in sample repository."
    )
    st.stop()

st.subheader(f"Repository: {st.session_state.get('label', analysis.root.name)}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Python files", len(analysis.python_files))
col2.metric("Symbols", len(analysis.symbols))
col3.metric("Skipped", len(analysis.skipped))
col4.metric("Warnings", len(analysis.issues))

tree_tab, skipped_tab, warnings_tab, symbols_tab, search_tab = st.tabs(
    ["File tree", "Skipped", "Warnings", "Symbols", "Search"]
)

with tree_tab:
    st.caption("Scanned files only. Anything missing here is listed in the Skipped tab.")
    st.code(analysis.tree_text(), language=None)

with skipped_tab:
    if not analysis.skipped:
        st.success("Nothing was skipped.")
    else:
        reasons = Counter(item.reason for item in analysis.skipped)
        summary = "; ".join(f"{r} ({n})" for r, n in reasons.items())
        st.caption("Why things were skipped: " + summary)
        st.dataframe(
            [{"path": i.path, "type": i.kind, "reason": i.reason} for i in analysis.skipped],
            hide_index=True,
        )

with warnings_tab:
    if not analysis.issues:
        st.success("No warnings. Every Python file was parsed.")
    else:
        st.dataframe(
            [{"file": i.path, "warning": i.message} for i in analysis.issues], hide_index=True
        )

with symbols_tab:
    kinds = analysis.index.counts_by_kind()
    st.caption(", ".join(f"{kind}: {n}" for kind, n in sorted(kinds.items())) or "No symbols.")
    st.dataframe(
        [
            {
                "symbol": s.qualname,
                "kind": s.kind,
                "location": s.location,
                "docstring": (s.docstring.splitlines() or [""])[0],
            }
            for s in analysis.symbols
        ],
        hide_index=True,
    )

with search_tab:
    with st.form("search_form"):
        query = st.text_input("Keyword(s)", placeholder="e.g. validate_email or discount percent")
        fields = st.multiselect("Search in", list(SEARCH_FIELDS), default=list(SEARCH_FIELDS))
        limit = st.number_input("Max results", min_value=1, max_value=100, value=20)
        submitted = st.form_submit_button("Search")
    if submitted:
        try:
            hits = analysis.index.search(query, fields=fields, limit=int(limit))
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.write(f"{len(hits)} result(s)")
            for hit in hits:
                sym = hit.symbol
                st.markdown(
                    f"`{sym.qualname}` · {sym.kind} · `{sym.location}` · "
                    f"score {hit.score} · matched in {', '.join(hit.matched)}"
                )
                with st.expander("Show source"):
                    st.code(sym.code, language="python")