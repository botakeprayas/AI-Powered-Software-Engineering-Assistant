"""Streamlit chat page (Phase 3): ask questions about an indexed repository.

Needs a local Ollama server, or the MOCK provider. Nothing leaves this computer unless
OLLAMA_BASE_URL points at another machine (the sidebar warns you).
"""

import logging
import sys
from pathlib import Path

# Pages run on their own, so add the project root (app/pages/1_Chat.py -> two levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.chat.models import STATUS_ERROR, Answer, RepoInfo  # noqa: E402
from core.chat.qa import QAEngine  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402

st.set_page_config(page_title="Repository chat", page_icon="💬", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("chat_page")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)  # loaded once, reused across reruns


def close_engine() -> None:
    engine = st.session_state.pop("chat_engine", None)
    if engine is not None:
        try:
            engine.close()
        except Exception:
            logger.warning("Could not close the previous index", exc_info=True)


def index_repository(folder_text: str) -> None:
    engine = None
    try:
        root = validate_folder(folder_text)
        with st.spinner("Indexing... (the first run loads the embedding model)"):
            embedder = get_embedder(embedding_model_name())
            engine = CodeSearchEngine(ScannedSymbolSource(root), embedder, default_db_path(root))
            stats = engine.sync()
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
            chat_engine=engine, chat_name=root.name, chat_symbols=stats.total, chat_messages=[]
        )
        return
    if engine is not None:
        engine.close()


def render_answer(answer: Answer) -> None:
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
                st.code(sym.code, language="python")
    if answer.raw_text and answer.warnings:
        with st.expander("Raw model output (not accepted as an answer)"):
            st.text(answer.raw_text)


st.title("Repository chat")
st.caption("Phase 3: ask about your code. Claims cite file and line range.")

with st.sidebar:
    st.header("Repository")
    analysis = st.session_state.get("analysis")  # set by the Phase 1 page, if used
    default_folder = str(getattr(analysis, "root", settings.sample_repo_dir))
    folder_text = st.text_input("Folder path", value=default_folder)
    if st.button("Index repository", type="primary"):
        index_repository(folder_text)
    st.header("Language model")
    use_mock = st.checkbox("Use MOCK provider (no model needed)", value=False)

try:
    llm = create_llm_from_env(provider_override="mock" if use_mock else None)
    qa_config = load_qa_config()
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

st.sidebar.caption(f"Provider: {llm.provider} · Model: {llm.model_name}")
if llm.is_mock:
    st.sidebar.warning("MOCK provider: answers are canned text, not from a real model.")
elif not llm.is_local:
    st.sidebar.warning(f"Code excerpts are sent to {llm.endpoint}, not kept on this computer.")
if st.sidebar.button("Check LLM"):
    status = llm.check_ready()
    if status.ok:
        st.sidebar.success(status.message)
    else:
        st.sidebar.error(status.message)
if st.sidebar.button("Clear conversation"):
    st.session_state["chat_messages"] = []

engine = st.session_state.get("chat_engine")
if engine is None:
    st.info("Choose a folder in the sidebar and press **Index repository** to start.")
    st.stop()

st.caption(
    f"Repository: {st.session_state['chat_name']} · "
    f"{st.session_state['chat_symbols']} symbols indexed"
)
messages = st.session_state.setdefault("chat_messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message["answer"])

question = st.chat_input("Ask a question about this repository")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    repo = RepoInfo(st.session_state["chat_name"], st.session_state["chat_symbols"])
    qa = QAEngine(engine, llm, qa_config, repo)
    with st.chat_message("assistant"):
        with st.spinner("Searching the code and asking the model..."):
            answer = qa.ask_safe(question)
        render_answer(answer)
    messages.append({"role": "assistant", "answer": answer})