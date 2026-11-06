"""Detect added, modified, deleted and renamed files between two manifests.

The file comparison is the ground truth. Git's rename report is only a hint, used when it
agrees with that comparison. The classification of a rename changes the report only: the old
path's data is removed and the new path is processed either way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.memory.gitrepo import GitRepo
from core.memory.models import SOURCE_CONTENT, SOURCE_GIT, FileChanges, FileState, RenameInfo


def detect_changes(
    stored: Mapping[str, FileState],
    current: Mapping[str, FileState],
    hints: Sequence[RenameInfo] = (),
) -> FileChanges:
    """Compare the indexed manifest (`stored`) with the current one."""
    both = set(stored) & set(current)
    modified = sorted(p for p in both if stored[p].content_hash != current[p].content_hash)
    unchanged = sorted(p for p in both if stored[p].content_hash == current[p].content_hash)
    added = set(current) - set(stored)
    deleted = set(stored) - set(current)
    renamed: list[RenameInfo] = []

    # 1. Git's own rename report, only where it agrees with the file comparison.
    for hint in sorted(hints, key=lambda h: (h.old_path, h.new_path)):
        if hint.old_path in deleted and hint.new_path in added:
            renamed.append(RenameInfo(hint.old_path, hint.new_path, hint.similarity, SOURCE_GIT))
            deleted.discard(hint.old_path)
            added.discard(hint.new_path)

    # 2. Identical bytes under a new path: a rename (or a move) found by content alone.
    by_hash: dict[str, list[str]] = {}
    for path in sorted(deleted):
        by_hash.setdefault(stored[path].content_hash, []).append(path)
    for path in sorted(added):
        candidates = by_hash.get(current[path].content_hash)
        if candidates:
            old_path = candidates.pop(0)
            renamed.append(RenameInfo(old_path, path, 100, SOURCE_CONTENT))
            deleted.discard(old_path)
            added.discard(path)

    renamed.sort(key=lambda r: (r.new_path, r.old_path))
    return FileChanges(sorted(added), modified, sorted(deleted), renamed, unchanged)


def git_renames(
    git: GitRepo, old_commit: str | None, new_commit: str | None = None
) -> tuple[list[RenameInfo], str]:
    """Renames git reports between `old_commit` and `new_commit` (or the working tree).

    Returns (renames, note). The note explains why git could not help, or is empty.
    """
    if not old_commit:
        return [], "There is no indexed commit, so git could not report renames."
    if not git.available():
        return [], "git is not available for this folder, so renames use identical content."
    if not git.commit_exists(old_commit):
        return [], (
            f"The indexed commit {old_commit[:12]} is not in this repository (history was "
            "rewritten, or it is a different repository); renames use identical content."
        )
    changes = git.changed_files(old_commit, new_commit)
    if changes is None:
        return [], "git diff failed, so renames use identical content."
    renames = [
        RenameInfo(c.old_path, c.path, c.similarity, SOURCE_GIT)
        for c in changes
        if c.status == "R" and c.old_path
    ]
    return renames, ""