"""Semantic-search configuration read from environment variables / .env."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from core.settings import DEFAULT_ENV_FILE, PROJECT_ROOT
from services.embedding_service import DEFAULT_MODEL


def embedding_model_name() -> str:
    load_dotenv(DEFAULT_ENV_FILE, override=False)
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def index_dir() -> Path:
    load_dotenv(DEFAULT_ENV_FILE, override=False)
    path = Path(os.getenv("INDEX_DIR", "data/indexes"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_db_path(repo: Path) -> Path:
    """One index file per repository path, so different repos never overwrite each other."""
    digest = hashlib.sha256(str(Path(repo).resolve()).lower().encode("utf-8")).hexdigest()
    return index_dir() / f"semantic-{digest[:10]}.db"