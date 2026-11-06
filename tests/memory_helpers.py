"""Helpers for the Phase 8 tests: real throwaway git repositories built from impact_sample."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from impact_helpers import IMPACT_SAMPLE

GIT_FLAGS = [
    "-c", "user.name=tester", "-c", "user.email=tester@example.com",
    "-c", "commit.gpgsign=false",
]
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *GIT_FLAGS, *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init")
    git(root, "config", "core.autocrlf", "false")


def commit_all(root: Path, message: str = "change") -> str:
    """Stage everything, commit, and return the new HEAD hash."""
    git(root, "add", "-A")
    git(root, "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} was not found in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def copy_sample(dest: Path) -> Path:
    shutil.copytree(IMPACT_SAMPLE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def sample_repo(tmp_path: Path) -> tuple[Path, str]:
    """A git repository holding a copy of data/impact_sample, with one commit."""
    root = copy_sample(tmp_path / "repo")
    init_repo(root)
    return root, commit_all(root, "initial")