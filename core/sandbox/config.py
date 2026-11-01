"""Sandbox settings read from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.settings import DEFAULT_ENV_FILE

TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class SandboxConfig:
    timeout_seconds: float = 60.0
    max_output_bytes: int = 200 * 1024
    max_copy_bytes: int = 50 * 1024 * 1024
    max_files: int = 3000
    allow_loopback: bool = False  # off: even localhost sockets are refused
    python: str = ""  # empty: use the interpreter that runs the app

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("SANDBOX_TIMEOUT_SECONDS must be greater than 0.")
        if self.max_output_bytes < 1000:
            raise ValueError("SANDBOX_MAX_OUTPUT_KB must be at least 1.")
        if self.max_copy_bytes < 1024:
            raise ValueError("SANDBOX_MAX_COPY_MB must be at least 1.")
        if self.max_files < 1:
            raise ValueError("SANDBOX_MAX_FILES must be at least 1.")


def _number(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    word = raw.strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be on or off, got {raw!r}.")


def load_sandbox_config(env_file: Path | None = DEFAULT_ENV_FILE) -> SandboxConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    return SandboxConfig(
        timeout_seconds=_number("SANDBOX_TIMEOUT_SECONDS", 60.0, float),
        max_output_bytes=_number("SANDBOX_MAX_OUTPUT_KB", 200, int) * 1024,
        max_copy_bytes=_number("SANDBOX_MAX_COPY_MB", 50, int) * 1024 * 1024,
        max_files=_number("SANDBOX_MAX_FILES", 3000, int),
        allow_loopback=_flag("SANDBOX_ALLOW_LOOPBACK", False),
    )