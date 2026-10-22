"""Bridge: lets the Phase 2 search engine use this scanner as its symbol source.

Phase 2 is NOT changed by this file. To switch later, scripts/semantic_search.py would use
ScannedSymbolSource(repo) instead of AstSymbolSource(repo). Note that this scanner labels
async code "async_function" / "async_method", so those symbols would be re-embedded once.
"""

from __future__ import annotations

from pathlib import Path

from core.repo.pipeline import analyze_repository
from core.semantic.models import Symbol


class ScannedSymbolSource:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.errors: list[str] = []  # same attribute name as AstSymbolSource

    def load(self) -> list[Symbol]:
        analysis = analyze_repository(self.root)
        try:
            self.errors = [f"{issue.path}: {issue.message}" for issue in analysis.issues]
            return analysis.symbols
        finally:
            analysis.close()