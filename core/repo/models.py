"""Plain data types for the scanner, extractor and search.

The Symbol type is shared with the semantic-search code (core/semantic/models.py), so one
definition is used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.semantic.models import Symbol


@dataclass(frozen=True)
class RepoFile:
    """A file that passed the scanner's filters."""

    path: str  # relative to the repo root, forward slashes
    size: int
    is_python: bool


@dataclass(frozen=True)
class SkippedItem:
    """A file or folder the scanner refused, with the reason."""

    path: str
    kind: str  # "dir" or "file"
    reason: str


@dataclass(frozen=True)
class FileIssue:
    """A warning about one file (syntax error, bad encoding, empty file, ...)."""

    path: str
    message: str


@dataclass
class ScanResult:
    root: Path
    files: list[RepoFile] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)
    limit_reached: bool = False


@dataclass(frozen=True)
class KeywordHit:
    symbol: Symbol
    score: int
    matched: tuple[str, ...]  # which fields matched: name, docstring, code