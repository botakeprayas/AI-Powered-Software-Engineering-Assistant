"""Application settings loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class Settings:
    """Immutable settings object. Contains no secrets in Phase 0."""

    app_env: str
    log_level: str
    data_dir: Path
    sample_repo_dir: Path


def _resolve_dir(raw: str) -> Path:
    """Treat relative paths as relative to the project root, not the current directory."""
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env_file: Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Load settings. Real environment variables win over values in the .env file.

    Pass env_file=None to skip reading .env (useful in tests).
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)  # silently does nothing if the file is missing

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {VALID_LOG_LEVELS}, got {log_level!r}")

    data_dir = _resolve_dir(os.getenv("DATA_DIR", "data"))
    return Settings(
        app_env=os.getenv("APP_ENV", "development").strip().lower(),
        log_level=log_level,
        data_dir=data_dir,
        sample_repo_dir=data_dir / "sample_repo",
    )