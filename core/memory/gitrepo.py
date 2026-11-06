"""A thin, read-only wrapper over the git command line.

Why subprocess and not GitPython: no new dependency, GitPython also runs the git
executable underneath, and it has known handle leaks on Windows. We need about eight
read-only commands with machine-readable output. Every call uses an argument list (never a
shell) with a timeout, and a revision is accepted only if it is a hex commit hash, so it can
never be read as an option. The repository is never modified.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BLOCKED_ENV = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"})
REVISION = re.compile(r"[0-9a-fA-F]{7,64}")
FULL_HASH = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


@dataclass(frozen=True)
class GitChange:
    status: str  # A added, M modified, D deleted, R renamed, C copied, T type change
    path: str  # the new path for R and C
    old_path: str | None = None
    similarity: int | None = None  # 0-100 for R and C


@dataclass(frozen=True)
class CommitInfo:
    commit: str
    committed_at: str  # ISO 8601
    subject: str


class GitRepo:
    """Git facts about the folder `root` (which may be a subfolder of a repository)."""

    def __init__(self, root: Path, timeout: float = 30.0) -> None:
        self.root = Path(root).resolve()
        self._timeout = timeout

    def _run(self, *args: str) -> subprocess.CompletedProcess[bytes] | None:
        env = {k: v for k, v in os.environ.items() if k not in BLOCKED_ENV}
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"  # a read-only query must not take locks
        try:
            return subprocess.run(
                ["git", "-c", "core.quotepath=false", *args],
                cwd=self.root, env=env, capture_output=True, timeout=self._timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _text(proc: subprocess.CompletedProcess[bytes]) -> str:
        return proc.stdout.decode("utf-8", "replace")

    def available(self) -> bool:
        """True when git exists and the folder is inside a git work tree."""
        proc = self._run("rev-parse", "--is-inside-work-tree")
        return bool(proc and proc.returncode == 0 and self._text(proc).strip() == "true")

    def head(self) -> str | None:
        """The full hash of HEAD, or None (no git, or a repository without commits)."""
        proc = self._run("rev-parse", "--verify", "--quiet", "HEAD")
        if proc is None or proc.returncode != 0:
            return None
        value = self._text(proc).strip()
        return value if FULL_HASH.fullmatch(value) else None

    def branch(self) -> str | None:
        """The current branch name, or None (detached HEAD, or no git)."""
        proc = self._run("symbolic-ref", "--short", "--quiet", "HEAD")
        if proc is None or proc.returncode != 0:
            return None
        return self._text(proc).strip() or None

    def commit_exists(self, revision: str) -> bool:
        if not REVISION.fullmatch(revision):
            return False
        proc = self._run("cat-file", "-e", f"{revision}^{{commit}}")
        return bool(proc and proc.returncode == 0)

    def is_dirty(self) -> bool | None:
        """Uncommitted or untracked files inside this folder? None when git can't say."""
        proc = self._run("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", ".")
        if proc is None or proc.returncode != 0:
            return None
        return bool(proc.stdout.strip(b"\0"))

    def changed_files(self, old: str, new: str | None = None) -> list[GitChange] | None:
        """Files changed between commit `old` and commit `new` (or the working tree).

        Paths are relative to this folder, and only files inside it are reported. Returns
        None when git could not answer, which is different from an empty list.
        """
        revisions = [old] if new is None else [old, new]
        if not all(REVISION.fullmatch(rev) for rev in revisions):
            return None
        proc = self._run(
            "diff", "--name-status", "-M", "-z", "--relative", *revisions, "--", "."
        )
        if proc is None or proc.returncode != 0:
            return None
        tokens = self._text(proc).split("\0")
        found: list[GitChange] = []
        index = 0
        while index < len(tokens):
            status = tokens[index]
            index += 1
            if not status:
                continue
            if status[0] in "RC":
                if index + 1 >= len(tokens):
                    break
                old_path, new_path = tokens[index], tokens[index + 1]
                index += 2
                similarity = int(status[1:]) if status[1:].isdigit() else None
                found.append(GitChange(status[0], new_path, old_path, similarity))
            else:
                if index >= len(tokens):
                    break
                found.append(GitChange(status[0], tokens[index]))
                index += 1
        return found

    def file_at(self, commit: str, rel_path: str) -> str | None:
        """The text of a file as it was in `commit`, or None."""
        parts = rel_path.replace("\\", "/").split("/")
        unsafe = (
            not REVISION.fullmatch(commit)
            or not rel_path
            or "\0" in rel_path
            or "\n" in rel_path
            or rel_path.startswith(("/", "-"))
            or ".." in parts
        )
        if unsafe:
            return None
        proc = self._run("show", f"{commit}:./{'/'.join(parts)}")
        if proc is None or proc.returncode != 0:
            return None
        return self._text(proc)

    def commit_info(self, commit: str) -> CommitInfo | None:
        if not REVISION.fullmatch(commit):
            return None
        proc = self._run("log", "-1", "--format=%H%x00%cI%x00%s", commit, "--")
        if proc is None or proc.returncode != 0:
            return None
        parts = self._text(proc).strip("\n").split("\x00")
        if len(parts) < 3:
            return None
        return CommitInfo(parts[0], parts[1], parts[2])