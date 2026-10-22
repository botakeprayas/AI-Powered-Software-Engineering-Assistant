"""Pipeline: scan -> extract -> index. The one entry point used by the UI and the scripts."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.repo.extractor import extract_file
from core.repo.index import SymbolIndex
from core.repo.models import FileIssue, RepoFile, ScanResult, SkippedItem
from core.repo.scanner import scan_repository
from core.repo.tree import render_tree
from core.semantic.models import Symbol

logger = logging.getLogger(__name__)

STATUS_NOTES = {
    "empty": "[empty file]",
    "syntax_error": "[syntax error]",
    "encoding_error": "[bad encoding]",
    "error": "[unreadable]",
}


@dataclass
class RepoAnalysis:
    root: Path
    scan: ScanResult
    symbols: list[Symbol]
    issues: list[FileIssue]
    skipped: list[SkippedItem]
    file_status: dict[str, str]  # path -> parsed | empty | syntax_error | ... | not_parsed
    index: SymbolIndex

    @property
    def python_files(self) -> list[RepoFile]:
        return [f for f in self.scan.files if f.is_python]

    def tree_annotations(self) -> dict[str, str]:
        counts = Counter(symbol.file for symbol in self.symbols)
        notes: dict[str, str] = {}
        for path, status in self.file_status.items():
            if status == "parsed":
                n = counts.get(path, 0)
                notes[path] = f"[{n} symbol{'' if n == 1 else 's'}]"
            elif status in STATUS_NOTES:
                notes[path] = STATUS_NOTES[status]
        return notes

    def tree_text(self) -> str:
        paths = [f.path for f in self.scan.files]
        return render_tree(paths, self.tree_annotations(), root_name=self.root.name)

    def close(self) -> None:
        self.index.close()


def analyze_repository(root: Path, extra_skipped: Sequence[SkippedItem] = ()) -> RepoAnalysis:
    """Scan a folder, extract symbols from its Python files and build the SQLite index.

    Problem files become warnings; they never raise.
    """
    scan = scan_repository(root)
    symbols: list[Symbol] = []
    issues: list[FileIssue] = []
    status: dict[str, str] = {}
    for repo_file in scan.files:
        if not repo_file.is_python:
            status[repo_file.path] = "not_parsed"
            continue
        extraction = extract_file(scan.root, repo_file)
        status[repo_file.path] = extraction.status
        symbols.extend(extraction.symbols)
        issues.extend(extraction.issues)
    index = SymbolIndex()
    index.replace_all(symbols)
    logger.info("Analysis done: %d symbols, %d warnings", len(symbols), len(issues))
    return RepoAnalysis(
        root=scan.root,
        scan=scan,
        symbols=symbols,
        issues=issues,
        skipped=[*extra_skipped, *scan.skipped],
        file_status=status,
        index=index,
    )