"""Verify the Phase 0 development environment.

Run from the project root:  python scripts/verify_env.py
Exit code 0 = all checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 10)
# import name -> pip package name
REQUIRED_IMPORTS = {"streamlit": "streamlit", "pytest": "pytest", "dotenv": "python-dotenv"}


def check_python() -> tuple[bool, str]:
    current = sys.version_info[:3]
    ok = current >= MIN_PYTHON
    return ok, f"{'.'.join(map(str, current))} (need >= {'.'.join(map(str, MIN_PYTHON))})"


def check_venv() -> tuple[bool, str]:
    active = sys.prefix != sys.base_prefix
    if active:
        return True, sys.prefix
    return False, "no virtual environment active; run .venv\\Scripts\\Activate.ps1"


def check_import(module: str, package: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, f"{package} {metadata.version(package)}"
    except Exception as exc:  # noqa: BLE001 - report any failure, do not crash
        return False, f"{package} failed to import: {exc}"


def check_ruff() -> tuple[bool, str]:
    # Ruff is a command-line tool, so we run it instead of importing it.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return False, f"could not run ruff: {exc}"
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, "ruff is not installed in this environment"


def check_env_ignored() -> tuple[bool, str]:
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        return False, ".gitignore is missing"
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True, ".env is ignored by Git (confirmed with git check-ignore)"
        if result.returncode == 1:
            return False, ".env is NOT ignored by Git; add '.env' to .gitignore"
        # Other codes (e.g. 128) usually mean "not a Git repository yet": fall through.
    except FileNotFoundError:
        pass  # Git is not installed or not on PATH: fall through.
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    if ".env" in lines:
        return True, ".gitignore lists .env (text check only; run 'git init' for a full check)"
    return False, ".gitignore does not list .env"


def check_env_example() -> tuple[bool, str]:
    ok = (PROJECT_ROOT / ".env.example").exists()
    return ok, ".env.example found" if ok else ".env.example is missing"


def main() -> int:
    checks: list[tuple[str, tuple[bool, str]]] = [
        ("Python version", check_python()),
        ("Virtual environment", check_venv()),
    ]
    for module, package in REQUIRED_IMPORTS.items():
        checks.append((f"Import {module}", check_import(module, package)))
    checks += [
        ("Ruff", check_ruff()),
        (".env example file", check_env_example()),
        (".env git-ignored", check_env_ignored()),
    ]

    print("Phase 0 environment check")
    print("=" * 25)
    failures = 0
    for name, (ok, detail) in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += not ok

    print()
    if failures:
        print(f"{failures} check(s) failed. See the messages above.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())