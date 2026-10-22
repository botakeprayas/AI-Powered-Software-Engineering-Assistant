"""Plain data types shared by the semantic-search modules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TEXT_FORMAT_VERSION = 1  # bump when embedding_text() changes so old vectors are recomputed
MAX_CODE_CHARS = 1500  # MiniLM reads only ~256 tokens, so more code is wasted
MAX_DOC_CHARS = 600

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def humanize(identifier: str) -> str:
    """'ShoppingCart.add_item' -> 'shopping cart add item'."""
    spaced = _CAMEL.sub(" ", identifier)
    return " ".join(word.lower() for word in _WORD_SPLIT.split(spaced) if word)


@dataclass(frozen=True)
class Symbol:
    """One function, method or class, with its exact location in the source."""

    key: str  # unique id, e.g. "billing.py::ShoppingCart.add_item"
    file: str  # path relative to the repo root, with forward slashes
    kind: str  # "function", "method" or "class"
    name: str
    qualname: str
    start_line: int  # 1-based, includes decorators
    end_line: int  # 1-based, inclusive
    docstring: str
    code: str

    @property
    def location(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """Text sent to the embedding model: name first, then docstring, then code."""
        lines = [f"{self.kind} {self.qualname} ({humanize(self.qualname)})"]
        if self.docstring:
            lines.append(f"Description: {self.docstring[:MAX_DOC_CHARS]}")
        lines.append(f"File: {self.file}")
        lines.append("Code:")
        lines.append(self.code[:MAX_CODE_CHARS])
        return "\n".join(lines)


@dataclass(frozen=True)
class SearchResult:
    symbol: Symbol
    score: float  # cosine (semantic), BM25 (keyword) or RRF (hybrid)
    semantic_rank: int | None = None
    semantic_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None


@dataclass
class SyncStats:
    total: int = 0
    embedded: int = 0
    reused: int = 0
    removed: int = 0
    embedded_files: list[str] = field(default_factory=list)
    seconds: float = 0.0