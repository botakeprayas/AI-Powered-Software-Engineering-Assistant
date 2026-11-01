"""Run pytest for a repository inside a disposable sandbox (a temporary copy).

Examples (from the project root):
  python scripts\\sandbox_run.py --repo . --test tests/test_settings.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.sandbox.config import load_sandbox_config  # noqa: E402
from core.sandbox.runner import Sandbox, SandboxError  # noqa: E402
from core.settings import load_settings  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tests in a disposable sandbox")
    parser.add_argument("--repo", required=True, help="repository folder (it is only copied)")
    parser.add_argument("--test", action="append", required=True, help="test file or id")
    parser.add_argument("--timeout", type=float, default=None, help="seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        config = load_sandbox_config()
        if args.timeout is not None:
            config = dataclasses.replace(config, timeout_seconds=args.timeout)
        with Sandbox(root, config) as box:
            report = box.copy_report
            print(f"Sandbox copy: {box.repo}")
            print(f"Copied {report.files} file(s), {report.bytes:,} bytes. "
                  f"Not copied: {len(report.skipped)}")
            for item in report.skipped[:10]:
                print(f"  not copied: {item.path} ({item.reason})")
            result = box.run_pytest(args.test)
    except (RepoLoadError, SandboxError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary.text if result.summary else "no summary"
    print(f"\nStatus: {result.status} | pytest exit code: {result.returncode} | {summary}")
    print(f"Duration: {result.duration:.1f}s | timed out: {result.timed_out}"
          f" | output cut: {result.truncated}")
    for test_id, outcome in sorted(result.outcomes.items()):
        print(f"  {outcome}  {test_id}")
    tail = "\n".join(result.stdout.splitlines()[-40:])
    print("\n--- stdout (last lines) ---\n" + tail)
    if result.stderr.strip():
        print("\n--- stderr ---\n" + result.stderr[-2000:])
    print("\nThese results come from real pytest output inside the sandbox copy.")
    return 0 if result.status == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())