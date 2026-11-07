"""Streamlit page (Phase 8): commit-aware repository memory.

Every answer records the repository version it was based on. If the repository moved on,
the page says so, and the stale policy decides what happens.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.chat.models import STATUS_ERROR  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.memory.config import STALE_POLICIES, load_memory_config  # noqa: E402
from core.memory.freshness import BEHIND, FRESH, NOT_INDEXED, STALE, Freshness  # noqa: E402
from core.memory.render import (  # noqa: E402
    citation_rows,
    history_rows,
    record_markdown,
    stamp_text,
)
from core.memory.service import MemoryService, VersionedAnswer  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.semantic.config import embedding_model_name  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402
from services.embedding_service import SentenceTransformerEmbedder  # noqa: E402

st.set_page_config(page_title="Repository memory", page_icon="🧠", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("memory_page")

POLICY_TEXT = {
    "update": "Update the index first, then answer (recommended)",
    "warn": "Answer from the old version, with a loud warning",
    "block": "Refuse to answer until the index is updated",
}


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


def open_service(folder_text: str, memory_config) -> None:
    try:
        root = validate_folder(folder_text)
        previous = st.session_state.pop("mem_service", None)
        if previous is not None:
            previous.close()
        embedder = get_embedder(embedding_model_name())
        service = MemoryService(root, embedder=embedder, config=memory_config)
        with st.spinner("Checking the index (the first run loads the embedding model)..."):
            if service.freshness().status == NOT_INDEXED:
                service.refresh(note="first index from the page")
            fresh = service.freshness()
    except RepoLoadError as exc:
        st.sidebar.error(str(exc))
    except Exception:
        logger.exception("Opening the repository failed")
        st.sidebar.error("Opening failed unexpectedly. Details are in the terminal log.")
    else:
        st.session_state.update(
            mem_service=service, mem_fresh=fresh, mem_messages=[], mem_update=""
        )


def show_freshness(fresh: Freshness) -> None:
    text = f"Index status: {fresh.status.upper()}. {fresh.headline}"
    if fresh.status == FRESH:
        st.success(text)
    elif fresh.status == BEHIND:
        st.info(text)
    elif fresh.status == STALE:
        st.error(text)
    else:
        st.warning(text)
    for reason in fresh.reasons:
        st.caption(reason)


def show_answer(item: VersionedAnswer) -> None:
    if item.blocked:
        st.error(item.reason)
        return
    if item.cached is not None:
        st.info(f"Stored answer #{item.cached.id} reused: same indexed content and settings.")
        st.markdown(record_markdown(item.cached))
    else:
        answer = item.validated.answer
        if answer.status == STATUS_ERROR:
            st.error(answer.message)
            return
        st.markdown(answer_to_markdown(answer))
        policy = item.validated.report.policy
        if policy is not None and policy.level != "not_applicable":
            st.caption(f"Evidence level: {policy.level} (rule-based, not a probability)")
    when = item.freshness.status if item.freshness else "unknown"
    st.caption(f"Based on {stamp_text(item.stamp)} · index was {when} · history #{item.record_id}")
    if item.citations:
        st.dataframe(citation_rows(item.citations), hide_index=True)
    for warning in item.warnings:
        st.warning(warning)


def show_history(service: MemoryService) -> None:
    records = service.history(50)
    st.subheader("Answer history")
    if not records:
        st.caption("No stored answers yet.")
        return
    st.dataframe(history_rows(records), hide_index=True)
    ids = [r.id for r in records]
    chosen = st.selectbox("Open a stored answer", ids, key="mem_open")
    record = next(r for r in records if r.id == chosen)
    if record.outdated:
        st.warning(record.outdated_reason)
    else:
        st.success("This answer is based on the content that is indexed now.")
    st.caption(f"Question: {record.question} · based on {stamp_text(record.version)}")
    st.markdown(record_markdown(record))
    if record.citations:
        st.dataframe(citation_rows(record.citations), hide_index=True)


st.title("Repository memory")
st.caption("Phase 8: every answer records the repository version it was based on.")

try:
    memory_config = load_memory_config()
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.data_dir / "impact_sample"))
    if st.button("Load repository", type="primary"):
        open_service(folder_text, memory_config)
    st.header("When the index is stale")
    default = STALE_POLICIES.index(memory_config.stale_policy)
    policy = st.selectbox(
        "Policy", STALE_POLICIES, index=default, format_func=POLICY_TEXT.get, key="mem_policy"
    )
    st.header("Language model")
    use_mock = st.checkbox("Use MOCK provider (no model needed)", value=False)
    reuse = st.checkbox("Reuse a stored answer when the indexed content is identical", value=False)

service = st.session_state.get("mem_service")
if service is None:
    st.info("Choose a folder in the sidebar and press **Load repository**.")
    st.stop()

try:
    llm = create_llm_from_env(provider_override="mock" if use_mock else None)
    qa_config = load_qa_config()
    validation_config = load_validation_config()
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

st.sidebar.caption(f"Provider: {llm.provider} · Model: {llm.model_name}")
if llm.is_mock:
    st.sidebar.warning("MOCK provider: answers are canned text, not from a real model.")
st.caption(f"Repository: {service.root}")
left, right = st.columns(2)
if left.button("Check freshness"):
    st.session_state["mem_fresh"] = service.freshness()
if right.button("Update index now"):
    with st.spinner("Updating the index (only changed files are reprocessed)..."):
        update = service.refresh(note="manual update from the page")
    st.session_state["mem_update"] = update.describe()
    st.session_state["mem_fresh"] = service.freshness()
show_freshness(st.session_state["mem_fresh"])
if st.session_state.get("mem_update"):
    st.code(st.session_state["mem_update"], language=None)

messages = st.session_state.setdefault("mem_messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message["answer"])

question = st.chat_input("Ask a question about this repository")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the index, searching and answering..."):
            try:
                item = service.ask(
                    question, llm, qa_config=qa_config, validation_config=validation_config,
                    policy=policy, reuse=reuse,
                )
            except Exception:
                logger.exception("Answering failed")
                st.error("Answering failed unexpectedly. Details are in the terminal log.")
                st.stop()
        show_answer(item)
    messages.append({"role": "assistant", "answer": item})
    if item.freshness is not None:
        st.session_state["mem_fresh"] = item.freshness

show_history(service)