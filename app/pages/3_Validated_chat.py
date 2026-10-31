"""Streamlit page (Phase 5): chat with answer validation and an evidence panel.

Levels shown here come from fixed rules in the configuration. They are not probabilities.
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
from core.chat.models import STATUS_ERROR, RepoInfo  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402
from core.validation.engine import ValidatedAnswer, ValidatingQAEngine  # noqa: E402
from core.validation.models import (  # noqa: E402
    LEVEL_DISCLAIMER,
    LEVEL_INSUFFICIENT,
    LEVEL_LIMITED,
    LEVEL_NOT_APPLICABLE,
    LEVEL_SUFFICIENT,
    UNSUPPORTED,
    WEAK,
)
from core.validation.render import FLAG_TEXT  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402

st.set_page_config(page_title="Validated chat", page_icon="✅", layout="wide")

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("validated_chat_page")


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


def close_engine() -> None:
    engine = st.session_state.pop("vchat_engine", None)
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
            vchat_engine=engine, vchat_root=root, vchat_name=root.name,
            vchat_symbols=stats.total, vchat_messages=[],
        )
        return
    if engine is not None:
        engine.close()


def show_level(validated: ValidatedAnswer) -> None:
    report = validated.report
    if validated.validation_error:
        st.error(validated.validation_error)
    policy = report.policy
    if policy is None or policy.level == LEVEL_NOT_APPLICABLE:
        return
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


def show_flags(validated: ValidatedAnswer) -> None:
    for check in validated.report.claims:
        if check.final not in (UNSUPPORTED, WEAK):
            continue
        title = "Unsupported conclusion" if check.final == UNSUPPORTED else "Weakly supported"
        st.warning(f"{title} ({check.label}, {check.kind}): {check.text}")
        for reason in check.reasons:
            st.caption(reason)
        if check.judge_reason:
            st.caption(f"judge: {check.judge_reason}")
    for note in validated.report.notes:
        st.caption(f"Note: {note}")


def show_panel(validated: ValidatedAnswer) -> None:
    answer, report = validated.answer, validated.report
    with st.expander("Evidence inspection"):
        st.caption(
            "Scores are retrieval similarity and BM25 values. "
            "They are not the probability that an answer is correct."
        )
        st.markdown("**Retrieved candidates**")
        if validated.candidates:
            st.dataframe(validated.candidates, hide_index=True)
        else:
            st.caption("Nothing was retrieved.")
        st.markdown("**Evidence the model was shown**")
        if validated.evidence:
            st.dataframe(validated.evidence, hide_index=True)
        else:
            st.caption("No excerpts were sent to the model.")
        if report.claims:
            st.markdown("**Claim checks**")
            st.dataframe(
                [
                    {
                        "claim": c.label,
                        "kind": c.kind,
                        "verdict": FLAG_TEXT.get(c.final, c.final),
                        "basis": c.basis,
                        "rules": c.rule_verdict,
                        "judge": c.judge_verdict,
                        "cites": ", ".join(c.evidence_ids),
                        "text": c.text,
                    }
                    for c in report.claims
                ],
                hide_index=True,
            )
        for item in answer.citations():
            sym = item.symbol
            st.markdown(f"**{item.id}** · `{item.location}` · {sym.kind} `{sym.qualname}`")
            st.code(sym.code, language="python")
        if report.judge_model:
            st.caption(f"Judge model: {report.judge_model}")


def render_validated(validated: ValidatedAnswer) -> None:
    answer = validated.answer
    if answer.status == STATUS_ERROR:
        st.error(answer.message)
        return
    st.markdown(answer_to_markdown(answer))
    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")
    show_level(validated)
    show_flags(validated)
    show_panel(validated)


st.title("Validated chat")
st.caption("Phase 5: every citation is re-checked against the indexed repository.")

with st.sidebar:
    st.header("Repository")
    folder_text = st.text_input("Folder path", value=str(settings.sample_repo_dir))
    if st.button("Index repository", type="primary"):
        index_repository(folder_text)
    st.header("Language model")
    use_mock = st.checkbox("Use MOCK provider (no model needed)", value=False)
    use_judge = st.checkbox("Use LLM judge (one extra model call)", value=True)

try:
    llm = create_llm_from_env(provider_override="mock" if use_mock else None)
    qa_config = load_qa_config()
    config = dataclasses.replace(load_validation_config(), judge_enabled=use_judge)
except ValueError as exc:
    st.sidebar.error(f"Configuration problem in .env: {exc}")
    st.stop()

st.sidebar.caption(f"Provider: {llm.provider} · Model: {llm.model_name}")
if llm.is_mock:
    st.sidebar.warning("MOCK provider: answers and judge verdicts are not from a real model.")
elif not llm.is_local:
    st.sidebar.warning(f"Code excerpts are sent to {llm.endpoint}, not kept on this computer.")
if st.sidebar.button("Check LLM"):
    status = llm.check_ready()
    (st.sidebar.success if status.ok else st.sidebar.error)(status.message)
if st.sidebar.button("Clear conversation"):
    st.session_state["vchat_messages"] = []

engine = st.session_state.get("vchat_engine")
if engine is None:
    st.info("Choose a folder in the sidebar and press **Index repository** to start.")
    st.stop()

st.caption(
    f"Repository: {st.session_state['vchat_name']} · "
    f"{st.session_state['vchat_symbols']} symbols indexed"
)
messages = st.session_state.setdefault("vchat_messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_validated(message["validated"])

question = st.chat_input("Ask a question about this repository")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    repo = RepoInfo(st.session_state["vchat_name"], st.session_state["vchat_symbols"])
    validating = ValidatingQAEngine(
        engine, llm, st.session_state["vchat_root"], qa_config, config, repo
    )
    with st.chat_message("assistant"):
        with st.spinner("Searching, answering and validating..."):
            validated = validating.ask(question)
        render_validated(validated)
    messages.append({"role": "assistant", "validated": validated})