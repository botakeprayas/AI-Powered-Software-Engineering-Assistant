"""Validation settings read from environment variables / .env."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.settings import DEFAULT_ENV_FILE, PROJECT_ROOT

TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class ValidationConfig:
    """Evidence policy and checker settings. Every value is a rule threshold."""

    min_valid_citations: int = 1  # cited excerpts that must pass the citation check
    min_strong_evidence: int = 1  # of those, how many must also pass the score test
    min_cosine: float = 0.30  # strong if similarity is at least this ...
    min_keyword_score: float = 8.0  # ... or keyword rank <= 3 with BM25 at least this
    min_supported_fraction: float = 0.5  # share of claims that must be supported
    max_unsupported_facts: int = 0
    fact_anchor_coverage: float = 0.8  # share of named code items found in cited code
    inference_anchor_coverage: float = 0.5
    word_coverage: float = 0.4  # fallback when a claim names no code items
    judge_enabled: bool = True
    judge_max_claims: int = 10
    judge_max_code_chars: int = 1500
    log_enabled: bool = True
    log_path: Path = PROJECT_ROOT / "data" / "logs" / "evaluations.jsonl"
    log_code: bool = False  # off: the log keeps code hashes and lengths, not code


# (field, environment variable, cast, minimum, maximum)
NUMERIC_SETTINGS = (
    ("min_valid_citations", "VALIDATION_MIN_VALID_CITATIONS", int, 0, None),
    ("min_strong_evidence", "VALIDATION_MIN_STRONG_EVIDENCE", int, 0, None),
    ("min_cosine", "VALIDATION_MIN_COSINE", float, -1.0, 1.0),
    ("min_keyword_score", "VALIDATION_MIN_KEYWORD_SCORE", float, 0.0, None),
    ("min_supported_fraction", "VALIDATION_MIN_SUPPORTED_FRACTION", float, 0.0, 1.0),
    ("max_unsupported_facts", "VALIDATION_MAX_UNSUPPORTED_FACTS", int, 0, None),
    ("fact_anchor_coverage", "VALIDATION_FACT_ANCHOR_COVERAGE", float, 0.0, 1.0),
    ("inference_anchor_coverage", "VALIDATION_INFERENCE_ANCHOR_COVERAGE", float, 0.0, 1.0),
    ("word_coverage", "VALIDATION_WORD_COVERAGE", float, 0.0, 1.0),
    ("judge_max_claims", "VALIDATION_JUDGE_MAX_CLAIMS", int, 1, None),
)
FLAG_SETTINGS = (
    ("judge_enabled", "VALIDATION_JUDGE"),
    ("log_enabled", "EVAL_LOG_ENABLED"),
    ("log_code", "EVAL_LOG_CODE"),
)


def _number(name, default, cast, low, high):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if value < low:
        raise ValueError(f"{name} must be at least {low}, got {value}.")
    if high is not None and value > high:
        raise ValueError(f"{name} must be at most {high}, got {value}.")
    return value


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


def load_validation_config(env_file: Path | None = DEFAULT_ENV_FILE) -> ValidationConfig:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    base = ValidationConfig()
    values: dict[str, object] = {}
    for field_name, env_name, cast, low, high in NUMERIC_SETTINGS:
        values[field_name] = _number(env_name, getattr(base, field_name), cast, low, high)
    for field_name, env_name in FLAG_SETTINGS:
        values[field_name] = _flag(env_name, getattr(base, field_name))
    raw_path = (os.getenv("EVAL_LOG_PATH") or "").strip()
    log_path = Path(raw_path) if raw_path else base.log_path
    values["log_path"] = log_path if log_path.is_absolute() else PROJECT_ROOT / log_path
    return dataclasses.replace(base, **values)