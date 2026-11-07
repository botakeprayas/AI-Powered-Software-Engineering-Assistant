"""Is the index still a true picture of the repository? (Phase 8)

fresh        the indexed content, HEAD and the working-tree state all match
behind       HEAD or the dirty state moved, but the indexed Python content is identical,
             so the evidence is still valid (for example a docs-only commit)
stale        the indexed content differs from the files: answers may use outdated code
not_indexed  nothing has been indexed yet

The check reads and hashes the kept Python files. It parses nothing and writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.memory.changes import detect_changes, git_renames
from core.memory.gitrepo import GitRepo
from core.memory.hashing import tree_digest
from core.memory.manifest import build_manifest
from core.memory.models import FileChanges, VersionRecord
from core.memory.store import MemoryStore

FRESH = "fresh"
BEHIND = "behind"
STALE = "stale"
NOT_INDEXED = "not_indexed"
MAX_LISTED = 5


@dataclass(frozen=True)
class Freshness:
    status: str
    indexed: VersionRecord | None = None
    head: str | None = None
    dirty: bool = False
    tree_digest: str = ""
    changes: FileChanges | None = None
    reasons: tuple[str, ...] = ()

    @property
    def is_fresh(self) -> bool:
        return self.status == FRESH

    @property
    def headline(self) -> str:
        label = self.indexed.label if self.indexed else "?"
        if self.status == FRESH:
            return f"The index matches the repository (version {label})."
        if self.status == BEHIND:
            return (
                f"HEAD or the working-tree state differs from version {label}, but the "
                "indexed Python content is identical, so the evidence is still valid."
            )
        if self.status == STALE:
            return (
                f"The repository has changed since version {label} was indexed, so "
                "answers may use outdated code."
            )
        return "The repository has not been indexed yet."


def _listed(changes: FileChanges) -> str:
    renamed = [f"{r.old_path} -> {r.new_path}" for r in changes.renamed]
    parts = []
    for title, items in (
        ("added", changes.added),
        ("modified", changes.modified),
        ("deleted", changes.deleted),
        ("renamed", renamed),
    ):
        if items:
            more = " ..." if len(items) > MAX_LISTED else ""
            parts.append(f"{title}: {', '.join(items[:MAX_LISTED])}{more}")
    return "; ".join(parts)


def _git_reasons(
    git: GitRepo, version: VersionRecord, head: str | None, dirty: bool
) -> list[str]:
    reasons: list[str] = []
    if version.commit_hash and head and head != version.commit_hash:
        info = git.commit_info(head)
        tail = f" Latest commit: {info.subject}" if info else ""
        reasons.append(
            f"HEAD is {head[:12]} but the index was built at {version.commit_hash[:12]}.{tail}"
        )
    elif version.commit_hash is None and head:
        reasons.append(f"The index was built without a commit hash; HEAD is now {head[:12]}.")
    elif version.commit_hash and head is None:
        reasons.append(
            f"git is no longer available for this folder (the index was built at "
            f"{version.commit_hash[:12]})."
        )
    if dirty and not version.dirty:
        reasons.append("The working tree now has uncommitted changes.")
    elif not dirty and version.dirty:
        reasons.append(
            "The working tree no longer has the uncommitted changes that were indexed."
        )
    return reasons


def check_freshness(root: Path, store: MemoryStore, git: GitRepo | None = None) -> Freshness:
    root = Path(root).resolve()
    git = git if git is not None else GitRepo(root)
    version = store.current_version()
    stored = store.manifest()
    manifest = build_manifest(root)
    digest = tree_digest(manifest.files)
    git_ok = git.available()
    head = git.head() if git_ok else None
    dirty = bool(git.is_dirty()) if git_ok else False
    if version is None or not stored:
        reasons = ("Nothing has been indexed for this repository yet.",)
        return Freshness(NOT_INDEXED, None, head, dirty, digest, None, reasons)

    changes = detect_changes(stored, manifest.files)
    if changes.added and changes.deleted and version.commit_hash:
        hints, _ = git_renames(git, version.commit_hash)
        if hints:
            changes = detect_changes(stored, manifest.files, hints)
    reasons: list[str] = []
    content_changed = changes.has_changes
    if content_changed:
        reasons.append(
            f"Python files changed since indexing ({changes.summary()}). {_listed(changes)}"
        )
    elif digest != version.tree_digest:
        content_changed = True
        reasons.append(
            "The stored version record does not match the stored file list, so the index "
            "cannot be trusted."
        )
    git_reasons = _git_reasons(git, version, head, dirty)
    reasons += git_reasons
    if content_changed:
        status = STALE
    else:
        status = BEHIND if git_reasons else FRESH
    return Freshness(status, version, head, dirty, digest, changes, tuple(reasons))