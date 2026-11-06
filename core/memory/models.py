"""Plain data types for commit-aware repository memory (Phase 8)."""

from __future__ import annotations

from dataclasses import dataclass, field

SOURCE_GIT = "git"  # git reported the rename
SOURCE_CONTENT = "content"  # the file's bytes are identical to a deleted file's bytes


@dataclass(frozen=True)
class FileState:
    """What the index knows about one file: its path and the hash of its bytes."""

    path: str  # relative to the repository folder, forward slashes
    content_hash: str  # sha256 of the file's bytes
    size: int
    is_python: bool = True


@dataclass(frozen=True)
class RenameInfo:
    old_path: str
    new_path: str
    similarity: int | None = None  # 0-100
    source: str = SOURCE_CONTENT

    @property
    def exact(self) -> bool:
        return self.similarity == 100


@dataclass
class FileChanges:
    """Difference between the indexed manifest and the current one. Lists are sorted."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    renamed: list[RenameInfo] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def to_process(self) -> list[str]:
        """Paths whose content must be processed: added, modified and renamed-to paths."""
        return sorted({*self.added, *self.modified, *(r.new_path for r in self.renamed)})

    @property
    def to_remove(self) -> list[str]:
        """Paths whose stored data must go: deleted and renamed-from paths."""
        return sorted({*self.deleted, *(r.old_path for r in self.renamed)})

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted or self.renamed)

    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.modified)} modified, "
            f"{len(self.deleted)} deleted, {len(self.renamed)} renamed, "
            f"{len(self.unchanged)} unchanged"
        )


@dataclass(frozen=True)
class VersionRecord:
    """One indexed version of the repository."""

    id: int
    commit_hash: str | None  # None when the folder is not in a git repository
    dirty: bool  # the working tree had uncommitted changes when it was indexed
    tree_digest: str  # digest of the indexed file contents
    branch: str | None
    indexed_at: str
    file_count: int
    symbol_count: int
    parent_id: int | None
    note: str

    @property
    def label(self) -> str:
        """A short name shown in the UI and the logs."""
        if self.commit_hash:
            return self.commit_hash[:12] + ("+dirty" if self.dirty else "")
        return f"files:{self.tree_digest[:12]}"