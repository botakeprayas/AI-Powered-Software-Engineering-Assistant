import os

import pytest
from repo_helpers import SAMPLE_REPO

from core.sandbox.config import SandboxConfig
from core.sandbox.results import (
    STATUS_FAILED,
    STATUS_INCONCLUSIVE,
    STATUS_PASSED,
    STATUS_TIMEOUT,
)
from core.sandbox.runner import Sandbox, SandboxError

GUARD_TESTS = '''import os
import socket
import subprocess
import sys

import pytest

SECRET = __SECRET__
OUTSIDE = __OUTSIDE__


def test_reading_host_files_is_blocked():
    with pytest.raises(PermissionError, match="sandbox"):
        open(SECRET, encoding="utf-8").read()


def test_writing_outside_the_sandbox_is_blocked():
    with pytest.raises(PermissionError, match="sandbox"):
        open(OUTSIDE, "w", encoding="utf-8").write("x")


def test_network_is_blocked():
    with pytest.raises(PermissionError, match="sandbox"):
        socket.create_connection(("93.184.216.34", 80), timeout=2)


def test_starting_programs_is_blocked():
    with pytest.raises(PermissionError, match="sandbox"):
        subprocess.run([sys.executable, "-c", "print(1)"])


def test_host_environment_is_not_inherited():
    assert "AISE_TEST_SECRET" not in os.environ


def test_sensitive_files_were_not_copied():
    assert not os.path.exists(".env")


def test_writing_inside_the_sandbox_works():
    with open("created_inside.txt", "w", encoding="utf-8") as handle:
        handle.write("ok")
'''
GUARD_NAMES = [
    "test_reading_host_files_is_blocked",
    "test_writing_outside_the_sandbox_is_blocked",
    "test_network_is_blocked",
    "test_starting_programs_is_blocked",
    "test_host_environment_is_not_inherited",
    "test_sensitive_files_were_not_copied",
    "test_writing_inside_the_sandbox_works",
]


def make_repo(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def guard_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("guard")
    host = base / "host"
    host.mkdir()
    secret = host / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    outside = host / "created_by_test.txt"
    source = GUARD_TESTS.replace("__SECRET__", repr(str(secret)))
    source = source.replace("__OUTSIDE__", repr(str(outside)))
    repo = make_repo(base / "repo", {"tests/test_guard.py": source, ".env": "KEY=1\n"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AISE_TEST_SECRET", "do-not-leak")
        with Sandbox(repo, SandboxConfig(timeout_seconds=60)) as box:
            result = box.run_pytest(["tests/test_guard.py"])
    return result, repo, outside


@pytest.mark.parametrize("name", GUARD_NAMES)
def test_guard_checks_inside_the_sandbox(guard_run, name):
    result, _, _ = guard_run
    key = f"tests/test_guard.py::{name}"
    assert result.outcomes.get(key) == "PASSED", result.stdout[-1500:]


def test_guard_run_overall_and_host_is_untouched(guard_run):
    result, repo, outside = guard_run
    assert result.status == STATUS_PASSED, result.stdout[-1500:]
    assert result.summary.passed == len(GUARD_NAMES)
    assert not outside.exists()
    assert not (repo / "created_inside.txt").exists()  # it was written in the copy only


def test_a_passing_and_a_failing_test_give_real_results(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "tests/test_mix.py": "def test_ok():\n    assert 1 + 1 == 2\n\n\n"
                             "def test_bad():\n    assert 1 == 2\n",
    })
    with Sandbox(repo) as box:
        result = box.run_pytest(["tests/test_mix.py"])
    assert result.status == STATUS_FAILED
    assert (result.summary.passed, result.summary.failed) == (1, 1)
    assert result.outcomes["tests/test_mix.py::test_ok"] == "PASSED"
    assert result.outcomes["tests/test_mix.py::test_bad"] == "FAILED"
    assert "assert 1 == 2" in result.stdout
    assert result.returncode == 1


def test_a_test_that_imports_repository_code_passes(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "calc.py": "def add(a, b):\n    return a + b\n",
        "tests/test_calc.py": (
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        ),
    })
    with Sandbox(repo) as box:
        assert box.run_pytest(["tests/test_calc.py"]).status == STATUS_PASSED


def test_the_original_repository_is_never_modified(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "tests/test_write.py": "def test_w():\n    open('junk.txt', 'w').write('x')\n",
    })
    before = sorted(p.name for p in repo.rglob("*"))
    with Sandbox(repo) as box:
        box.run_pytest(["tests/test_write.py"])
        assert box.exists("junk.txt")
    assert sorted(p.name for p in repo.rglob("*")) == before


def test_a_runaway_test_is_killed_after_the_timeout(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "tests/test_slow.py": "import time\n\n\ndef test_slow():\n    time.sleep(120)\n",
    })
    box = Sandbox(repo, SandboxConfig(timeout_seconds=4))
    try:
        result = box.run_pytest(["tests/test_slow.py"])
    finally:
        box.cleanup()
    assert result.timed_out
    assert result.status == STATUS_TIMEOUT
    assert result.summary is None
    assert result.duration < 30
    assert not box.path.exists()


def test_exiting_without_a_summary_is_not_reported_as_passed(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "tests/test_exit.py": "import os\n\n\ndef test_exit():\n    os._exit(0)\n",
    })
    with Sandbox(repo) as box:
        result = box.run_pytest(["tests/test_exit.py"])
    assert result.status == STATUS_INCONCLUSIVE


def test_long_output_is_cut_but_the_summary_survives(tmp_path):
    repo = make_repo(tmp_path / "repo", {
        "tests/test_big.py": "def test_big():\n    assert False, 'x' * 60000\n",
    })
    with Sandbox(repo, SandboxConfig(max_output_bytes=3000)) as box:
        result = box.run_pytest(["tests/test_big.py"])
    assert result.truncated
    assert len(result.stdout) < 3600
    assert result.status == STATUS_FAILED


def test_bad_test_ids_are_refused_before_anything_runs(tmp_path):
    repo = make_repo(tmp_path / "repo", {"tests/test_a.py": "def test_a():\n    pass\n"})
    with Sandbox(repo) as box:
        with pytest.raises(SandboxError):
            box.run_pytest(["--collect-only"])
        with pytest.raises(SandboxError):
            box.run_pytest([])
        with pytest.raises(SandboxError):
            box.write_file("../escape.py", "x = 1")


def test_the_sample_repository_can_be_copied_and_cleaned_up():
    box = Sandbox(SAMPLE_REPO)
    try:
        assert box.exists("shop/cart.py")
        assert box.path.exists()
    finally:
        box.cleanup()
    assert not box.path.exists()
    assert os.path.exists(SAMPLE_REPO)