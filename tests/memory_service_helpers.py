"""Helpers for the Phase 8 service tests. HashEmbedder is a MOCK, not a real model."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import numpy as np

from core.chat.models import QAConfig
from core.memory.config import MemoryConfig
from core.memory.service import MemoryService
from core.memory.store import MemoryStore
from core.validation.config import ValidationConfig
from services.embedding_service import EmbeddingError, EmbeddingService

QUESTION = "How is sales tax applied to an amount?"
OPEN_GATE = QAConfig(min_cosine=-1.0)  # let any retrieved code through the evidence gate


class HashEmbedder(EmbeddingService):
    """Hashed bag-of-words vectors. Counts what it embeds and can be made to fail."""

    def __init__(self, model_name: str = "hash-embedder-1", dim: int = 64) -> None:
        self._model_name = model_name
        self._dim = dim
        self.texts_embedded = 0
        self.fail = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts):
        items = list(texts)
        if self.fail:
            raise EmbeddingError("simulated embedding failure")
        self.texts_embedded += len(items)
        out = np.zeros((len(items), self._dim), dtype=np.float32)
        for row, text in enumerate(items):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def validation_config(tmp_path: Path) -> ValidationConfig:
    """Rule-based claim checks only (no judge call) and a log file inside tmp_path."""
    return dataclasses.replace(
        ValidationConfig(), judge_enabled=False, log_enabled=True, log_path=tmp_path / "log.jsonl"
    )


def make_service(root, tmp_path, embedder=None, policy="update") -> MemoryService:
    return MemoryService(
        root,
        embedder=embedder or HashEmbedder(),
        store=MemoryStore(),
        semantic_db=tmp_path / "vectors.db",
        config=MemoryConfig(stale_policy=policy),
    )


def ask(service, llm, tmp_path, question=QUESTION, qa_config=OPEN_GATE, **kwargs):
    return service.ask(
        question, llm, qa_config=qa_config, validation_config=validation_config(tmp_path),
        **kwargs,
    )