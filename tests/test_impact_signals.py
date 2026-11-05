import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from core.impact.signals import CombinedSignals, GitSignals, NoSignals, RuffSignals

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
_OLD = datetime.now(timezone.utc) - timedelta(days=400)
OLD_DATE = _OLD.strftime("%Y-%m-%dT%H:%M:%S +0000")


def git(root, *args, date=None):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, env=env,
                   check=True, capture_output=True)


@pytest.fixture
def history(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "old", date=OLD_DATE)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "recent")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "recent 2")
    return tmp_path


@needs_git
def test_churn_counts_recent_commits_per_file(history):
    signals = GitSignals(history, days=180)
    assert signals.churn("a.py") == 1  # the 2020 commit is outside the window
    assert signals.churn("b.py") == 2
    assert signals.churn("missing.py") == 0
    assert signals.notes == []


@needs_git
def test_a_longer_window_counts_older_commits(history):
    assert GitSignals(history, days=800).churn("a.py") == 2


@needs_git
def test_a_folder_that_is_not_a_repository_gives_none_and_a_note(tmp_path):
    signals = GitSignals(tmp_path)
    assert signals.churn("a.py") is None
    assert any("not inside a Git repository" in note for note in signals.notes)


def test_ruff_findings_come_back_with_line_numbers(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def f(x):\n    unused = x * 2\n    return x\n", encoding="utf-8"
    )
    (tmp_path / "clean.py").write_text("def g(x):\n    return x\n", encoding="utf-8")
    signals = RuffSignals(tmp_path)
    found = signals.lint(["mod.py", "clean.py"])
    assert found is not None, signals.notes
    assert [(f.code, f.line) for f in found["mod.py"]] == [("F841", 2)]
    assert found["clean.py"] == []


def test_ruff_ignores_the_project_config_and_handles_no_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff]\nextend-exclude = ['mod.py']\n", encoding="utf-8"
    )
    (tmp_path / "mod.py").write_text("def f():\n    unused = 1\n", encoding="utf-8")
    found = RuffSignals(tmp_path).lint(["mod.py"])
    assert found is not None and len(found["mod.py"]) == 1
    assert RuffSignals(tmp_path).lint([]) == {}


def test_a_missing_ruff_gives_none_and_a_note(tmp_path):
    signals = RuffSignals(tmp_path, python="definitely-not-a-python-executable")
    assert signals.lint(["mod.py"]) is None
    assert any("could not be started" in note for note in signals.notes)


def test_signals_can_be_switched_off():
    off = CombinedSignals()
    assert off.churn("a.py") is None and off.lint(["a.py"]) is None
    assert off.notes == ["Git history was turned off.", "Ruff was turned off."]
    none = NoSignals()
    assert none.churn("a.py") is None and none.lint(["a.py"]) is None