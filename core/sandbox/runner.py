"""Disposable sandbox: a temporary copy of the repository and a restricted pytest runner.

What this does and does not protect against is explained in the Phase 6 README section.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.repo.models import SkippedItem
from core.repo.scanner import scan_repository
from core.sandbox import sandbox_guard
from core.sandbox.config import SandboxConfig
from core.sandbox.results import RunResult, parse_outcomes, parse_summary

BOOTSTRAP = (
    "import sys\n"
    "guard_dir = sys.argv[1]\n"
    "sys.path.insert(0, guard_dir)\n"
    "import sandbox_guard\n"
    "sandbox_guard.install(guard_dir + '/policy.json')\n"
    "sys.path.remove(guard_dir)\n"
    "import pytest\n"
    "raise SystemExit(pytest.main(sys.argv[2:]))\n"
)
PYTHON_FLAGS = ["-E", "-s", "-B", "-X", "utf8"]  # ignore PYTHON* variables, user site, bytecode
PROXY = "http://127.0.0.1:9"  # a dead proxy: defense in depth next to the guard


class SandboxError(RuntimeError):
    """A sandbox problem, phrased for the user."""


@dataclass
class CopyReport:
    files: int
    bytes: int
    skipped: list[SkippedItem] = field(default_factory=list)  # never copied, with reasons


def check_node_id(node_id: str) -> str:
    """Accept only 'path/to/test_file.py' or 'path::Class::test'. No options, no escapes."""
    text = node_id.strip().replace("\\", "/")
    bad = (not text) or len(text) > 300 or "\n" in text or "\r" in text or text.startswith("-")
    path_part = text.split("::", 1)[0]
    if bad or not re.fullmatch(r"[\w./-]+", path_part):
        raise SandboxError(f"Not a valid test id: {node_id!r}")
    if path_part.startswith("/") or ".." in path_part.split("/"):
        raise SandboxError(f"Test paths must stay inside the repository: {node_id!r}")
    return text


def copy_repository(root: Path, dest: Path, config: SandboxConfig) -> CopyReport:
    """Copy only the files the Phase 1 scanner keeps (no .env, keys, venvs, .git, caches)."""
    try:
        scan = scan_repository(Path(root), max_files=config.max_files)
    except NotADirectoryError as exc:
        raise SandboxError(str(exc)) from exc
    if scan.limit_reached:
        raise SandboxError(f"The repository has more than {config.max_files:,} files.")
    dest = Path(dest).resolve()
    total = 0
    for item in scan.files:
        total += item.size
        if total > config.max_copy_bytes:
            limit_mb = config.max_copy_bytes // (1024 * 1024)
            raise SandboxError(f"The repository is larger than {limit_mb} MB; refusing to copy.")
        target = (dest / item.path).resolve()
        if dest not in target.parents:
            raise SandboxError(f"Unsafe path in the repository: {item.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scan.root / item.path, target)
    return CopyReport(len(scan.files), total, list(scan.skipped))


def _read_capped(path: Path, limit: int) -> tuple[str, bool]:
    """Keep the head and the tail of a long output (pytest's summary is at the end)."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= limit:
            return handle.read().decode("utf-8", "replace"), False
        half = limit // 2
        head = handle.read(half)
        handle.seek(size - half)
        tail = handle.read(half)
    note = f"\n[... {size - limit:,} bytes omitted ...]\n".encode()
    return (head + note + tail).decode("utf-8", "replace"), True


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and its children (the venv launcher starts a second process)."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=15, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


class Sandbox:
    """A temporary folder holding a copy of the repository. Use it as a context manager."""

    def __init__(self, repo_root: Path, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._base = Path(tempfile.mkdtemp(prefix="ai_se_sandbox_")).resolve()
        self.repo = self._base / "repo"
        self.support = self._base / "support"
        self.home = self._base / "home"
        self.tmp = self._base / "tmp"
        self._runs = 0
        try:
            for folder in (self.repo, self.support, self.home, self.tmp):
                folder.mkdir()
            self.copy_report = copy_repository(Path(repo_root), self.repo, self.config)
            self._install_support_files()
        except BaseException:
            shutil.rmtree(self._base, ignore_errors=True)
            raise

    @property
    def path(self) -> Path:
        return self._base

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        for _ in range(3):  # Windows may need a moment to release files of a killed process
            shutil.rmtree(self._base, ignore_errors=True)
            if not self._base.exists():
                return
            time.sleep(0.5)

    def _install_support_files(self) -> None:
        shutil.copyfile(Path(sandbox_guard.__file__), self.support / "sandbox_guard.py")
        policy = {
            "sandbox_root": str(self._base),
            "allow_loopback": self.config.allow_loopback,
        }
        (self.support / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
        (self.support / "pytest_empty.ini").write_text("[pytest]\n", encoding="utf-8")

    # ---- files inside the copy ---------------------------------------------------------
    def _inside_repo(self, rel: str) -> Path:
        text = rel.replace("\\", "/")
        unsafe = not text or text.startswith("/") or ".." in text.split("/")
        unsafe = unsafe or re.match(r"[A-Za-z]:", text) is not None
        target = (self.repo / text).resolve() if not unsafe else None
        if target is None or self.repo not in target.parents:
            raise SandboxError(f"Unsafe path: {rel!r}")
        return target

    def write_file(self, rel: str, text: str) -> None:
        target = self._inside_repo(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")

    def read_text(self, rel: str) -> str:
        return self._inside_repo(rel).read_text(encoding="utf-8")

    def exists(self, rel: str) -> bool:
        return self._inside_repo(rel).exists()

    # ---- running pytest ----------------------------------------------------------------
    def _python(self) -> str:
        return self.config.python or sys.executable

    def _environment(self) -> dict[str, str]:
        system = os.environ.get("SYSTEMROOT", "")
        path_parts = [os.path.dirname(self._python())]
        if system:
            path_parts.append(os.path.join(system, "System32"))
        env = {
            "PATH": os.pathsep.join(path_parts),
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "APPDATA": str(self.home),
            "LOCALAPPDATA": str(self.home),
            "TEMP": str(self.tmp),
            "TMP": str(self.tmp),
            "TMPDIR": str(self.tmp),
            "USERNAME": "sandbox",
            "NO_PROXY": "*",
            "HTTP_PROXY": PROXY,
            "HTTPS_PROXY": PROXY,
            "ALL_PROXY": PROXY,
        }
        for key in ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _config_args(self) -> list[str]:
        """Use the repository's own pytest config if it has one, else an empty one."""
        for name, marker in (
            ("pytest.ini", None), ("pyproject.toml", "[tool.pytest"),
            ("tox.ini", "[pytest]"), ("setup.cfg", "[tool:pytest]"),
        ):
            path = self.repo / name
            if not path.is_file():
                continue
            if marker is None:
                return ["-c", str(path)]
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker in text:
                return ["-c", str(path)]
        return ["-c", str(self.support / "pytest_empty.ini")]

    def run_pytest(self, node_ids: list[str], timeout: float | None = None) -> RunResult:
        """Run the given test files or node ids inside the copy. Never raises for test failures."""
        ids = [check_node_id(item) for item in node_ids]
        if not ids:
            raise SandboxError("Give at least one test file or test id to run.")
        limit = timeout if timeout is not None else self.config.timeout_seconds
        pytest_args = [
            "-q", "--no-header", "--tb=short", "--color=no", "-rpfE", "-p", "no:cacheprovider",
            "--rootdir", str(self.repo), "--confcutdir", str(self.repo),
            *self._config_args(), *ids,
        ]
        python = self._python()
        command = [python, *PYTHON_FLAGS, "-c", BOOTSTRAP, str(self.support), *pytest_args]
        shown = [python, *PYTHON_FLAGS, "-c", "<sandbox bootstrap>", "<support>", *pytest_args]
        self._runs += 1
        out_path = self.support / f"run{self._runs}_stdout.txt"
        err_path = self.support / f"run{self._runs}_stderr.txt"
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        started = time.perf_counter()
        timed_out = False
        returncode: int | None = None
        try:
            with out_path.open("wb") as out, err_path.open("wb") as err:
                proc = subprocess.Popen(
                    command, cwd=self.repo, env=self._environment(),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kwargs,
                )
                try:
                    returncode = proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_tree(proc)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
        except OSError as exc:
            duration = time.perf_counter() - started
            message = f"Could not start the sandboxed process: {exc}"
            return RunResult(shown, None, False, duration, error=message)
        duration = time.perf_counter() - started
        stdout, cut_out = _read_capped(out_path, self.config.max_output_bytes)
        stderr, cut_err = _read_capped(err_path, self.config.max_output_bytes)
        return RunResult(
            command=shown,
            returncode=returncode,
            timed_out=timed_out,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
            truncated=cut_out or cut_err,
            summary=None if timed_out else parse_summary(stdout),
            outcomes={} if timed_out else parse_outcomes(stdout),
        )