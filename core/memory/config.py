"""Memory settings read from environment variables / .env (Phase 8)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.settings import DEFAULT_ENV_FILE

POLICY_UPDATE = "update"  # refresh the index, then answer
POLICY_WARN = "warn"  # answer from the old version with a loud warning
POLICY_BLOCK = "block"  # refuse to answer from a stale index
STALE_POLICIES = (POLICY_UPDATE, POLICY_WARN, POLICY_BLOCK)


@dataclass(frozen=True)
class MemoryConfig:
    stale_policy: str = POLICY_UPDATE
    retrieval_cache_size: int = 64

    def __post_init__(self) -> None:
        if self.stale_policy not in STALE_POLICIES:
            raise ValueError(
                f"MEMORY_STALE_POLICY must be one of {STALE_POLICIES}, got {self.stale_policy!r}."
            )
        if not 1 <= self.retrieval_cache_size <= 10_000:
            raise ValueError("MEMORY_RETRIEVAL_CACHE_SIZE must be between 1 and 10000.")


def load_memory_config(env_file: Path | None = DEFAULT_ENV_FILE) -> MemoryConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    base = MemoryConfig()
    policy = (os.getenv("MEMORY_STALE_POLICY") or base.stale_policy).strip().lower()
    size = base.retrieval_cache_size
    raw = os.getenv("MEMORY_RETRIEVAL_CACHE_SIZE")
    if raw is not None and raw.strip():
        try:
            size = int(raw.strip())
        except ValueError as exc:
            message = f"MEMORY_RETRIEVAL_CACHE_SIZE must be a whole number, got {raw!r}."
            raise ValueError(message) from exc
    return MemoryConfig(policy, size)