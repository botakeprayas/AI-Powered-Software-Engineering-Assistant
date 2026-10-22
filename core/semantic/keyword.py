"""Small BM25 keyword index.

FALLBACK / stand-in: this is NOT your Phase 1 keyword search. It exists so hybrid search can
run on its own. CodeSearchEngine accepts a custom keyword function to replace it.

BM25 in one sentence: a word counts more when it is rare across all symbols and appears
often in this symbol, with a penalty for very long symbols.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from core.semantic.models import Symbol

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
STOPWORDS = frozenset(
    {"a", "an", "the", "that", "this", "of", "to", "for", "in", "is", "and", "or",
     "with", "it", "be", "by", "on", "as"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; identifiers are kept whole AND split (snake_case, camelCase)."""
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        pieces = [p for p in _CAMEL.sub("_", ident).lower().split("_") if p]
        if len(pieces) > 1:
            tokens.append(ident.lower())
        tokens.extend(pieces)
    return [t for t in tokens if t not in STOPWORDS]


def _symbol_tokens(symbol: Symbol) -> list[str]:
    # The name is repeated 3x so a name match outweighs a match buried in the body.
    return (
        tokenize(symbol.qualname) * 3
        + tokenize(symbol.docstring)
        + tokenize(symbol.code)
        + tokenize(symbol.file)
    )


class KeywordIndex:
    def __init__(self, symbols: list[Symbol], k1: float = 1.5, b: float = 0.75) -> None:
        self._symbols = list(symbols)
        self._k1 = k1
        self._b = b
        docs = [_symbol_tokens(s) for s in self._symbols]
        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) for d in docs]
        self._avgdl = max(sum(self._len) / len(self._len), 1.0) if self._len else 1.0
        doc_freq: Counter[str] = Counter()
        for tf in self._tf:
            doc_freq.update(tf.keys())
        n = len(self._symbols)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in doc_freq.items()}

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Ranked (symbol_key, score). An exact name match is always placed first."""
        q_tokens = list(dict.fromkeys(tokenize(query)))
        scored: list[tuple[str, float]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for token in q_tokens:
                freq = tf.get(token)
                if not freq:
                    continue
                norm = freq + self._k1 * (1 - self._b + self._b * self._len[i] / self._avgdl)
                score += self._idf[token] * freq * (self._k1 + 1) / norm
            if score > 0:
                scored.append((self._symbols[i].key, score))
        scored.sort(key=lambda item: (-item[1], item[0]))

        needle = query.strip().lower()
        exact = [
            s.key
            for s in self._symbols
            if needle and needle in (s.name.lower(), s.qualname.lower())
        ]
        if exact:
            top = scored[0][1] if scored else 0.0
            rest = [(k, sc) for k, sc in scored if k not in exact]
            scored = [(k, top + 1.0) for k in exact] + rest
        return scored[:limit]