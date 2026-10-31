"""Helpers for the Phase 5 end-to-end tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from core.validation.config import ValidationConfig


class StubSearch:
    """MOCK search engine: it retrieves nothing and has an empty index."""

    def search(self, query, mode="hybrid", top_k=5):
        return []

    def stored_locations(self):
        return {}


class BrokenStore(StubSearch):
    """MOCK search engine whose index cannot be read (to test crash containment)."""

    def stored_locations(self):
        raise RuntimeError("index unavailable")


def config_for(tmp_path: Path, **changes) -> ValidationConfig:
    """Default config with the log pointed at a temp folder and switched off."""
    base = dataclasses.replace(
        ValidationConfig(), log_path=tmp_path / "logs" / "eval.jsonl", log_enabled=False
    )
    return dataclasses.replace(base, **changes)