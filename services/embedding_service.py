"""Embedding service interface and the Sentence Transformers implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or fails while embedding."""


class EmbeddingService(ABC):
    """Anything that turns text into vectors. Swap implementations without touching callers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model. It is part of every cache hash."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of each vector."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), L2-normalized."""


class SentenceTransformerEmbedder(EmbeddingService):
    """Local, free embedding model. Loaded lazily on first use (importing torch is slow)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed(["dimension probe"]).shape[1])
        return self._dimension

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc
        logger.info("Loading embedding model %s (the first run downloads it)", self._model_name)
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:  # noqa: BLE001 - many failure types; we explain them all
            raise EmbeddingError(
                f"Could not load embedding model {self._model_name!r}: {exc}\n"
                "Likely causes: no internet on the first run (the model is downloaded once "
                "from Hugging Face), a firewall or proxy, low disk space, or a misspelled "
                "model name. After one successful download it works offline."
            ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self._dimension or 0), dtype=np.float32)
        if not all(isinstance(t, str) for t in items):
            raise EmbeddingError("embed() expects a list of strings.")
        model = self._load()
        try:
            vectors = model.encode(
                items,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding failed for {len(items)} text(s): {exc}") from exc
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(items):
            raise EmbeddingError(f"Unexpected embedding shape {vectors.shape}.")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("The model returned NaN or infinite values.")
        self._dimension = int(vectors.shape[1])
        return vectors