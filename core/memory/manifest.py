"""The current manifest: content hashes of the Python files the Phase 1 scanner keeps.

Files the scanner skips (virtual environments, secrets, generated code, binaries) never
appear, and are never read. Non-Python files are not hashed: they change no symbol, no
embedding and no graph edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.memory.hashing import file_digest
from core.memory.models import FileState
from core.repo.models import SkippedItem
from core.repo.scanner import scan_repository


@dataclass
class ManifestResult:
    files: dict[str, FileState] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)


def build_manifest(root: Path) -> ManifestResult:
    """Scan `root` and hash every kept Python file. Unreadable files become warnings."""
    scan = scan_repository(Path(root))
    result = ManifestResult(skipped=list(scan.skipped))
    if scan.limit_reached:
        result.warnings.append(
            "The scanner's file limit was reached, so some files are not in the manifest."
        )
    for item in scan.files:
        if not item.is_python:
            continue
        try:
            data = (scan.root / item.path).read_bytes()
        except OSError as exc:
            reason = exc.strerror or str(exc)
            result.warnings.append(f"{item.path}: could not be read ({reason}); not indexed")
            continue
        result.files[item.path] = FileState(item.path, file_digest(data), len(data), True)
    return result