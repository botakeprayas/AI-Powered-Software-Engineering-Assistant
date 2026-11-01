import os

import pytest
from repo_helpers import build_hazard_repo

from core.sandbox import sandbox_guard
from core.sandbox.config import SandboxConfig, load_sandbox_config
from core.sandbox.results import (
    STATUS_ERROR,
    STATUS_FAILED,
    STATUS_INCONCLUSIVE,
    STATUS_NO_TESTS,
    STATUS_PASSED,
    STATUS_TIMEOUT,
    PytestSummary,
    RunResult,
    parse_outcomes,
    parse_summary,
)
from core.sandbox.runner import SandboxError, check_node_id, copy_repository

ENV_NAMES = (
    "SANDBOX_TIMEOUT_SECONDS", "SANDBOX_MAX_OUTPUT_KB", "SANDBOX_MAX_COPY_MB",
    "SANDBOX_MAX_FILES", "SANDBOX_ALLOW_LOOPBACK",
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("3 passed in 0.02s", (3, 0, 0)),
        ("1 failed, 2 passed in 0.10s", (2, 1, 0)),
        ("2 passed, 1 warning in 0.5s", (2, 0, 0)),
        ("1 error in 0.1s", (0, 0, 1)),
        ("=== 1 failed, 1 passed in 0.05s ===", (1, 1, 0)),
    ],
)
def test_summary_line(line, expected):
    text = "some output\n" + line + "\n"
    summary = parse_summary(text)
    assert (summary.passed, summary.failed, summary.errors) == expected
    assert summary.seconds is not None


def test_no_tests_and_missing_summary():
    assert parse_summary("no tests ran in 0.01s").no_tests
    assert parse_summary("nothing useful here") is None
    assert parse_summary("") is None


def test_outcomes_come_from_the_last_summary_section():
    text = (
        "PASSED fake/line.py::not_real\n"
        "=== short test summary info ===\n"
        "PASSED tests/test_a.py::test_ok\n"
        "FAILED tests/test_a.py::test_bad - assert 1 == 2\n"
        "ERROR tests/test_b.py::test_fixture\n"
        "1 failed, 1 passed, 1 error in 0.2s\n"
    )
    assert parse_outcomes(text) == {
        "tests/test_a.py::test_ok": "PASSED",
        "tests/test_a.py::test_bad": "FAILED",
        "tests/test_b.py::test_fixture": "ERROR",
    }
    assert parse_outcomes("no summary") == {}


def make_result(returncode, summary=None, timed_out=False, error=""):
    return RunResult(["py"], returncode, timed_out, 1.0, summary=summary, error=error)


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (make_result(0, PytestSummary(passed=2)), STATUS_PASSED),
        (make_result(1, PytestSummary(passed=1, failed=1)), STATUS_FAILED),
        (make_result(1, PytestSummary(errors=1)), STATUS_FAILED),
        (make_result(0, None), STATUS_INCONCLUSIVE),
        (make_result(0, PytestSummary(skipped=1)), STATUS_INCONCLUSIVE),
        (make_result(5, PytestSummary(no_tests=True)), STATUS_NO_TESTS),
        (make_result(2, PytestSummary(errors=1)), STATUS_ERROR),
        (make_result(3, None), STATUS_ERROR),
        (make_result(None, None, timed_out=True), STATUS_TIMEOUT),
        (make_result(None, None, error="could not start"), STATUS_ERROR),
    ],
)
def test_status_classification(result, status):
    assert result.status == status


@pytest.mark.parametrize("good", ["tests/test_a.py", "tests\\test_a.py::Cls::test_x[1-2]", "a.py"])
def test_good_node_ids(good):
    assert check_node_id(good) == good.replace("\\", "/")


@pytest.mark.parametrize(
    "bad", ["", "--collect-only", "-x", "../other.py", "/abs/test.py", "a b.py", "t.py\nx"]
)
def test_bad_node_ids_are_rejected(bad):
    with pytest.raises(SandboxError):
        check_node_id(bad)


def test_copy_skips_secrets_and_junk(tmp_path):
    source = build_hazard_repo(tmp_path / "src")
    report = copy_repository(source, tmp_path / "dest", SandboxConfig())
    dest = tmp_path / "dest"
    assert (dest / "app.py").exists() and (dest / "shop" / "cart.py").exists()
    for hidden in (".env", ".env.local", "id_rsa", "keys/server.pem", ".venv", "node_modules"):
        assert not (dest / hidden).exists(), hidden
    reasons = {item.path: item.reason for item in report.skipped}
    assert "sensitive" in reasons[".env"]
    assert report.files > 0


def test_copy_limits(tmp_path):
    source = build_hazard_repo(tmp_path / "src")
    with pytest.raises(SandboxError, match="files"):
        copy_repository(source, tmp_path / "d1", SandboxConfig(max_files=2))
    with pytest.raises(SandboxError, match="larger"):
        copy_repository(source, tmp_path / "d2", SandboxConfig(max_copy_bytes=1024))
    with pytest.raises(SandboxError):
        copy_repository(tmp_path / "missing", tmp_path / "d3", SandboxConfig())


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_and_environment(clean_env, monkeypatch):
    assert load_sandbox_config(env_file=None) == SandboxConfig()
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SANDBOX_ALLOW_LOOPBACK", "on")
    config = load_sandbox_config(env_file=None)
    assert config.timeout_seconds == 5.0 and config.allow_loopback is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SANDBOX_TIMEOUT_SECONDS", "0"),
        ("SANDBOX_TIMEOUT_SECONDS", "soon"),
        ("SANDBOX_ALLOW_LOOPBACK", "maybe"),
        ("SANDBOX_MAX_FILES", "0"),
    ],
)
def test_config_rejects_bad_values(clean_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="SANDBOX_"):
        load_sandbox_config(env_file=None)


def make_policy(tmp_path, loopback=False):
    root = tmp_path / "box"
    root.mkdir()
    return root, sandbox_guard.build_policy(str(root), loopback)


def test_guard_allows_the_sandbox_and_the_python_install(tmp_path):
    root, policy = make_policy(tmp_path)
    assert sandbox_guard.decide("open", (str(root / "a.txt"), "r", 0), policy) is None
    assert sandbox_guard.decide("open", (str(root / "a.txt"), "w", 0), policy) is None
    assert sandbox_guard.decide("open", (os.__file__, "rb", 0), policy) is None
    assert sandbox_guard.decide("open", (os.devnull, "w", 0), policy) is None
    assert sandbox_guard.decide("open", (5, "r", 0), policy) is None


def test_guard_blocks_files_outside(tmp_path):
    root, policy = make_policy(tmp_path)
    other = tmp_path / "host.txt"
    other.write_text("secret", encoding="utf-8")
    assert "reading" in sandbox_guard.decide("open", (str(other), "r", 0), policy)
    assert "writing" in sandbox_guard.decide("open", (str(other), "w", 0), policy)
    flags = os.O_WRONLY | os.O_CREAT
    assert "writing" in sandbox_guard.decide("open", (str(other), "rb", flags), policy)
    assert "writing" in sandbox_guard.decide("os.remove", (str(other), None), policy)
    assert "reading" in sandbox_guard.decide("os.listdir", (str(tmp_path),), policy)
    assert sandbox_guard.decide("os.listdir", (str(root),), policy) is None


def test_guard_blocks_programs_links_and_network(tmp_path):
    _, policy = make_policy(tmp_path)
    for event in ("subprocess.Popen", "os.system", "os.symlink"):
        assert sandbox_guard.decide(event, (), policy)
    remote = (None, ("93.184.216.34", 80))
    assert "network" in sandbox_guard.decide("socket.connect", remote, policy)
    assert "network" in sandbox_guard.decide("socket.getaddrinfo", ("example.com", 80), policy)
    assert sandbox_guard.decide("some.other.event", (), policy) is None


def test_loopback_is_blocked_unless_allowed(tmp_path):
    local = (None, ("127.0.0.1", 8000))
    _, strict = make_policy(tmp_path / "a", loopback=False) if False else (None, None)
    root = tmp_path / "box"
    root.mkdir()
    strict = sandbox_guard.build_policy(str(root), False)
    relaxed = sandbox_guard.build_policy(str(root), True)
    assert sandbox_guard.decide("socket.connect", local, strict)
    assert sandbox_guard.decide("socket.connect", local, relaxed) is None
    remote = (None, ("93.184.216.34", 80))
    assert sandbox_guard.decide("socket.connect", remote, relaxed)