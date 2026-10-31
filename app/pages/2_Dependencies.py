"""Streamlit page (Phase 4): dependency graph of a Python repository.

Static analysis only: code is parsed, never run. Uncertain relationships are drawn dashed
and listed with their reasons. They are never guessed.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.graph.builder import build_graph  # noqa: E402
from core.graph.graph import DependencyGraph  # noqa: E402
from core.graph.models import (  # noqa: E402
    CONFIRMED,
    LIMITATIONS,
    NODE_CLASS,
    NODE_FUNCTION,
    UNCERTAIN,
    Relation,
)
from core.graph.viz import draw_view, select_view  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.pipeline import analyze_repository  # noqa: E402
from core.settings import load_settings  # noqa: E402

st.set_page_config(page_title="Dependencies", page_icon="🕸️", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("dependencies_page")
WHOLE_GRAPH = "(whole graph)"


def build_for(folder_text: str) -> None:
    try:
        root = validate_folder(folder_text)
        with st.spinner("Building the graph..."):
            analysis = analyze_repository(root)
            try:
                graph = build_graph(analysis)
            finally:
                analysis.close()
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except Exception:
        logger.exception("Graph build failed")
        st.sidebar.error("Building the graph failed. Details are in the terminal log.")
    else:
        st.session_state.update(dep_graph=graph, dep_root=str(root))


def relation_rows(graph: DependencyGraph, relations: list[Relation]) -> list[dict]:
    return [
        {
            "kind": r.kind,
            "certainty": r.certainty,
            "from": r.source_label,
            "to": r.target_label,
            "location": r.location,
            "reason": r.reason,
            "possible targets (not confirmed)": ", ".join(graph.label(c) for c in r.candidates),
        }
        for r in relations
    ]


st.title("Dependency graph")
st.caption("Phase 4: imports, definitions and calls, each traced to a file and line.")

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.data_dir / "graph_sample"))
    if st.button("Build graph", type="primary"):
        build_for(folder_text)

graph: DependencyGraph | None = st.session_state.get("dep_graph")
if graph is None:
    st.info("Choose a folder and press **Build graph**. The sample is `data/graph_sample`.")
    st.stop()

stats = graph.stats
cols = st.columns(4)
cols[0].metric("Files analyzed", stats.files_analyzed)
functions_and_classes = stats.nodes.get("function", 0) + stats.nodes.get("class", 0)
cols[1].metric("Functions and classes", functions_and_classes)
cols[2].metric("Confirmed calls", stats.edges.get("calls/confirmed", 0))
cols[3].metric("Uncertain calls", stats.edges.get("calls/uncertain", 0))
st.caption(
    "The graph was built when you pressed the button. Rebuild after editing code. "
    f"Calls not recorded (builtins, packages): {sum(stats.ignored.values())}."
)
with st.expander("What static analysis cannot resolve"):
    for line in LIMITATIONS:
        st.markdown(f"- {line}")

symbol_ids = sorted(
    node_id
    for node_id, data in graph.nx_graph.nodes(data=True)
    if data.get("kind") in (NODE_FUNCTION, NODE_CLASS)
)
choice = st.selectbox(
    "Focus on",
    [WHOLE_GRAPH, *symbol_ids],
    format_func=lambda n: n if n == WHOLE_GRAPH else f"{graph.label(n)}  [{graph.node(n)['file']}]",
)
left, middle, right = st.columns(3)
depth = left.slider("Depth around the focus", 1, 3, 1)
shown_kinds = middle.multiselect(
    "Show", [CONFIRMED, UNCERTAIN], default=[CONFIRMED, UNCERTAIN]
)
max_nodes = right.slider("Max boxes", 10, 80, 40)

focus = None if choice == WHOLE_GRAPH else choice
view = select_view(graph, focus, depth, tuple(shown_kinds), max_nodes)
st.pyplot(draw_view(graph, view))
if view.truncated:
    st.caption(f"Only {max_nodes} boxes are shown. Raise the limit or focus on one symbol.")

if focus is not None:
    st.subheader(graph.label(focus))
    tables = (
        ("Callers (confirmed)", graph.callers_of(focus)),
        ("Callees (confirmed)", graph.callees_of(focus)),
        ("Calls that could not be resolved", graph.uncertain_callees_of(focus)),
        ("Possible callers (uncertain, NOT confirmed)", graph.possible_callers_of(focus)),
    )
    for title, relations in tables:
        st.markdown(f"**{title}**")
        if relations:
            st.dataframe(relation_rows(graph, relations), hide_index=True)
        else:
            st.caption("none")

with st.expander("Every relationship"):
    every = graph.relations()
    st.dataframe(relation_rows(graph, every[:500]), hide_index=True)
    if len(every) > 500:
        st.caption(f"Showing the first 500 of {len(every)} relationships.")