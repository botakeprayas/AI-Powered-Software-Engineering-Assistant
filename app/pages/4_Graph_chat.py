"""Streamlit page (Phase 4): chat whose retrieved code is expanded with the call graph.

Needs a local Ollama server, or the MOCK provider.
"""

import dataclasses
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.chat.models import STATUS_ERROR, Answer, RepoInfo  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.graph.expand import (  # noqa: E402
    ExpansionResult,
    GraphEvidence,
    load_expansion_config,
    load_graph_context,
)
from core.graph.qa import GraphQAEngine  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402

st.set_page_config(page_title="Graph chat", page_icon="🔗", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("graph_chat_page")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


def close_engine() -> None:
    engine = st.session_state.pop("gchat_engine", None)
    if engine is not None:
        try:
            engine.close()
        except Exception:
            logger.warning("Could not close the previous index", exc_info=True)


def index_repository(folder_text: str) -> None:
    engine = None
    try:
        root = validate_folder(folder_text)
        with st.spinner("Indexing and building the graph..."):
            embedder = get_embedder(embedding_model_name())
            engine = CodeSearchEngine(ScannedSymbolSource(root), embedder, default_db_path(root))
            stats = engine.sync()
            context = load_graph_context(root)
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except EmbeddingError as exc:
        st.sidebar.error(f"Embedding model problem: {exc}")
    except Exception:
        logger.exception("Indexing failed")
        st.sidebar.error("Indexing failed unexpectedly. Details are in the terminal log.")
    else:
        close_engine()
        st.session_state.update(
            gchat_engine=engine, gchat_context=context, gchat_name=root.name,
            gchat_symbols=stats.total, gchat_messages=[],
        )
        return
    if engine is not None:
        engine.close()


def render_answer(answer: Answer, expansion: ExpansionResult) -> None:
    if answer.status == STATUS_ERROR:
        st.error(answer.message)
        return
    st.markdown(answer_to_markdown(answer))
    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")
    cited = answer.citations()
    if cited:
        with st.expander(f"Cited code ({len(cited)})"):
            for item in cited:
                sym = item.symbol
                st.markdown(f"**{item.id}** · `{item.location}` · {sym.kind} `{sym.qualname}`")
                if isinstance(item, GraphEvidence):
                    st.caption(f"Added by the call graph: {item.via}")
                st.code(sym.code, language="python")
    added = [e for e in answer.evidence if isinstance(e, GraphEvidence)]
    if added or expansion.notes or expansion.skipped:
        with st.expander(f"Call-graph context ({len(added)} added)"):
            for item in added:
                st.markdown(f"**{item.id}** `{item.location}` {item.symbol.qualname}")
                st.caption(item.via)
            if expansion.notes:
                st.markdown("**Notes (uncertain or module-level; not evidence)**")
                for note in expansion.notes:
                    st.caption(note)
            for reason in expansion.skipped:
                st.caption(f"Not added: {reason}")


st.title("Graph chat")
st.caption("Phase 4: retrieved code is expanded with confirmed callers and callees.")

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.data_dir / "graph_sample"))
    if st.button("Index repository", type="primary"):
        index_repository(folder_text)
    st.header("Language model")
    use_mock = st.checkbox("Use MOCK provider (no model needed)", value=False)

try:
    llm = create_llm_from_env(provider_override="mock" if use_mock else None)
    qa_config = load_qa_config()
    expansion_config = load_expansion_config()
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

depth = st.sidebar.slider("Call-graph depth", 1, 2, min(expansion_config.depth, 2))
expansion_config = dataclasses.replace(expansion_config, depth=depth)
st.sidebar.caption(f"Provider: {llm.provider} · Model: {llm.model_name}")
if llm.is_mock:
    st.sidebar.warning("MOCK provider: answers are canned text, not from a real model.")
elif not llm.is_local:
    st.sidebar.warning(f"Code excerpts are sent to {llm.endpoint}, not kept on this computer.")
if st.sidebar.button("Check LLM"):
    status = llm.check_ready()
    (st.sidebar.success if status.ok else st.sidebar.error)(status.message)
if st.sidebar.button("Clear conversation"):
    st.session_state["gchat_messages"] = []

engine = st.session_state.get("gchat_engine")
if engine is None:
    st.info("Choose a folder in the sidebar and press **Index repository** to start.")
    st.stop()

st.caption(
    f"Repository: {st.session_state['gchat_name']} · "
    f"{st.session_state['gchat_symbols']} symbols indexed"
)
messages = st.session_state.setdefault("gchat_messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message["answer"], message["expansion"])

question = st.chat_input("Ask a question that spans several files")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    repo = RepoInfo(st.session_state["gchat_name"], st.session_state["gchat_symbols"])
    qa = GraphQAEngine(
        engine, llm, st.session_state["gchat_context"], qa_config, repo, expansion_config
    )
    with st.chat_message("assistant"):
        with st.spinner("Searching, following the call graph and asking the model..."):
            answer = qa.ask_safe(question)
        render_answer(answer, qa.last_expansion)
    messages.append({"role": "assistant", "answer": answer, "expansion": qa.last_expansion})