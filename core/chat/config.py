"""Read LLM and QA settings from environment variables / .env."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from dotenv import load_dotenv

from core.chat.models import QAConfig
from core.settings import DEFAULT_ENV_FILE
from services.llm_service import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    LLMService,
    MockLLM,
    OllamaLLM,
)


def _env_number(name: str, default, cast, minimum):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def create_llm_from_env(
    provider_override: str | None = None, env_file: Path | None = DEFAULT_ENV_FILE
) -> LLMService:
    """Build the one LLM service. LLM_PROVIDER is 'ollama' (default) or 'mock'."""
    if env_file is not None:
        load_dotenv(env_file, override=False)
    provider = (provider_override or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if provider == "mock":
        return MockLLM()
    if provider != "ollama":
        raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'mock', got {provider!r}.")
    return OllamaLLM(
        base_url=(os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).strip(),
        model=(os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip(),
        timeout=_env_number("LLM_TIMEOUT_SECONDS", 120.0, float, 1.0),
        max_retries=_env_number("LLM_MAX_RETRIES", 2, int, 0),
        num_ctx=_env_number("LLM_NUM_CTX", 8192, int, 512),
    )


def load_qa_config(env_file: Path | None = DEFAULT_ENV_FILE) -> QAConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    base = QAConfig()
    max_evidence = _env_number("QA_MAX_EVIDENCE", base.max_evidence, int, 1)
    return dataclasses.replace(
        base,
        max_evidence=max_evidence,
        candidate_k=max(base.candidate_k, max_evidence + 3),
        min_cosine=_env_number("QA_MIN_COSINE", base.min_cosine, float, -1.0),
        min_keyword_score=_env_number("QA_MIN_KEYWORD_SCORE", base.min_keyword_score, float, 0.0),
    )