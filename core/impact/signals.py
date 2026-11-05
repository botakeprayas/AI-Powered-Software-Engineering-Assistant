"""Optional signals: Git churn and Ruff findings. Both fail soft: a missing tool or a failed
command becomes a note and the factor is left out of the ranking. Ruff runs in isolated mode
(default rules, no project config), reads the file as indexed, and never runs repository code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

GIT_ENV_BLOCKED = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"})


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    end_line: int
    code: str
    message: str


class NoSignals:
    """Use neither Git nor Ruff."""

    def __init__(self) -> None:
        self.notes = ["Git history and Ruff were not used."]

    def churn(self, path: str) -> int | None:
        return None

    def lint(self, paths: Sequence[str]) -> dict[str, list[Finding]] | None:
        return None


class GitSignals:
    """Commits that touched a file in the last `days` days (per file, not per function)."""

    def __init__(self, root: Path, days: int = 180, timeout: float = 30.0) -> None:
        self._root = Path(root)
        self._days = days
        self._timeout = timeout
        self._cache: dict[str, int | None] = {}
        self._inside: bool | None = None
        self.notes: list[str] = []

    def _run(self, *args: str) -> subprocess.CompletedProcess | None:
        env = {k: v for k, v in os.environ.items() if k not in GIT_ENV_BLOCKED}
        try:
            return subprocess.run(
                ["git", *args], cwd=self._root, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self._timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def available(self) -> bool:
        if self._inside is None:
            proc = self._run("rev-parse", "--is-inside-work-tree")
            self._inside = bool(proc and proc.returncode == 0 and proc.stdout.strip() == "true")
            if not self._inside:
                self.notes.append(
                    "Git history was not used: git is missing or this folder is not "
                    "inside a Git repository."
                )
        return self._inside

    def churn(self, path: str) -> int | None:
        if path in self._cache:
            return self._cache[path]
        count: int | None = None
        if self.available():
            proc = self._run("log", f"--since={self._days}.days.ago", "--format=%H", "--", path)
            if proc is not None and proc.returncode == 0:
                count = sum(1 for line in proc.stdout.splitlines() if line.strip())
            else:
                self.notes.append(f"git log failed for {path}, so its churn is unknown.")
        self._cache[path] = count
        return count


class RuffSignals:
    """Ruff findings (default rules) for the given files."""

    def __init__(self, root: Path, timeout: float = 60.0, python: str = "") -> None:
        self._root = Path(root)
        self._timeout = timeout
        self._python = python or sys.executable
        self.notes: list[str] = []

    def _relative(self, filename: object) -> str | None:
        try:
            return Path(str(filename)).resolve().relative_to(self._root.resolve()).as_posix()
        except (OSError, ValueError):
            return None

    def lint(self, paths: Sequence[str]) -> dict[str, list[Finding]] | None:
        files = sorted(set(paths))
        if not files:
            return {}
        command = [
            self._python, "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero",
            "--output-format=json", "--", *files,
        ]
        try:
            proc = subprocess.run(
                command, cwd=self._root, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self._timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self.notes.append("Ruff could not be started, so its findings are not scored.")
            return None
        if proc.returncode != 0:
            first = (proc.stderr.strip().splitlines() or ["unknown error"])[0][:200]
            self.notes.append(f"Ruff failed ({first}), so its findings are not scored.")
            return None
        try:
            items = json.loads(proc.stdout) if proc.stdout.strip() else []
        except json.JSONDecodeError:
            items = None
        if not isinstance(items, list):
            self.notes.append("Ruff printed output that could not be read; findings not scored.")
            return None
        per: dict[str, list[Finding]] = {name: [] for name in files}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = self._relative(item.get("filename"))
            if name not in per:
                continue
            start = item.get("location") or {}
            end = item.get("end_location") or {}
            row = int(start.get("row") or 0)
            per[name].append(
                Finding(
                    name, row, int(end.get("row") or row), str(item.get("code") or "syntax"),
                    str(item.get("message") or ""),
                )
            )
        return per


class CombinedSignals:
    """Git churn and Ruff findings together. Either part may be switched off."""

    def __init__(self, git: GitSignals | None = None, ruff: RuffSignals | None = None) -> None:
        self._git = git
        self._ruff = ruff

    @property
    def notes(self) -> list[str]:
        found: list[str] = []
        found += self._git.notes if self._git else ["Git history was turned off."]
        found += self._ruff.notes if self._ruff else ["Ruff was turned off."]
        return list(dict.fromkeys(found))

    def churn(self, path: str) -> int | None:
        return self._git.churn(path) if self._git else None

    def lint(self, paths: Sequence[str]) -> dict[str, list[Finding]] | None:
        return self._ruff.lint(paths) if self._ruff else None