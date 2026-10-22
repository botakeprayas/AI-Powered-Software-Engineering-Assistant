"""Shared fixtures: a small sample repository and embedding stand-ins."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from core.semantic.engine import CodeSearchEngine
from core.semantic.symbol_source import AstSymbolSource
from services.embedding_service import EmbeddingError, EmbeddingService, SentenceTransformerEmbedder

SAMPLE_FILES = {
    "auth.py": '''"""Account related helpers."""
import hashlib
import os


def sanitize_submission(fields):
    """Reject empty values and strip dangerous characters from data submitted by a user."""
    cleaned = {}
    for key, value in fields.items():
        text = str(value).strip()
        if not text:
            raise ValueError(f"{key} must not be empty")
        cleaned[key] = text.replace("<", "").replace(">", "")
    return cleaned


def salted_digest(text):
    """Scramble a plain text password with a random salt so it is stored unreadable."""
    salt = os.urandom(16)
    return salt + hashlib.pbkdf2_hmac("sha256", text.encode(), salt, 100_000)
''',
    "billing.py": '''"""Shopping and payment helpers."""


class ShoppingCart:
    """Holds the products a shopper intends to buy."""

    def __init__(self):
        self.items = []

    def add_item(self, name, price, quantity=1):
        """Put a product into the cart."""
        self.items.append((name, price, quantity))


def compute_invoice_total(items, tax_rate):
    """Add up the line items and apply sales tax to get the amount a customer owes."""
    subtotal = sum(price * quantity for _, price, quantity in items)
    return round(subtotal * (1 + tax_rate), 2)


def format_currency(amount):
    """Render a number as dollars with two decimal places."""
    return f"${amount:,.2f}"
''',
    "utils.py": '''"""General utilities."""
import functools
import json
import time


def log_call(func):
    """Decorator that prints the name of each function when it runs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper


@log_call
def read_json_config(path):
    """Load application settings from a JSON file on disk."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def retry_request(operation, attempts=3):
    """Call an unreliable network operation again after a failure, up to a limit."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError:
            time.sleep(2**attempt)
    raise RuntimeError("operation failed after retries")
''',
}
SAMPLE_SYMBOL_COUNT = 10


class CountingFakeEmbedder(EmbeddingService):
    """MOCK: hashed bag-of-words vectors. NOT semantic. Used only to test caching logic."""

    def __init__(self, model_name: str = "fake-model-1", dim: int = 64) -> None:
        self._model_name = model_name
        self._dim = dim
        self.texts_embedded = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts):
        items = list(texts)
        self.texts_embedded += len(items)
        out = np.zeros((len(items), self._dim), dtype=np.float32)
        for row, text in enumerate(items):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FailingEmbedder(CountingFakeEmbedder):
    """MOCK: always fails, to test that a failed embedding leaves the cache intact."""

    def embed(self, texts):
        raise EmbeddingError("simulated embedding failure")


def _write_files(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")


@pytest.fixture
def repo_dir(tmp_path):
    root = tmp_path / "repo"
    _write_files(root, SAMPLE_FILES)
    return root


@pytest.fixture
def fake_embedder():
    return CountingFakeEmbedder()


@pytest.fixture
def engine(repo_dir, fake_embedder, tmp_path):
    eng = CodeSearchEngine(AstSymbolSource(repo_dir), fake_embedder, tmp_path / "index.db")
    yield eng
    eng.close()


@pytest.fixture(scope="session")
def real_embedder():
    embedder = SentenceTransformerEmbedder()
    try:
        embedder.embed(["warm-up"])
    except EmbeddingError as exc:
        pytest.skip(f"Embedding model unavailable (these tests did NOT run): {exc}")
    return embedder


@pytest.fixture(scope="session")
def real_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("real_repo")
    _write_files(root, SAMPLE_FILES)
    return root


@pytest.fixture(scope="session")
def real_engine(real_embedder, real_repo, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("real_index") / "semantic.db"
    eng = CodeSearchEngine(AstSymbolSource(real_repo), real_embedder, db_path)
    eng.sync()
    yield eng
    eng.close()
    
@pytest.fixture
def make_fake_embedder():
    return CountingFakeEmbedder


@pytest.fixture
def make_failing_embedder():
    return FailingEmbedder