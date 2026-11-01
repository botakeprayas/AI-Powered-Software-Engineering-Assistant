"""Investigate a failure from a traceback; optionally verify the proposed fix in a sandbox.

Examples (from the project root):
  python scripts\\investigate.py --repo data\\bug_sample --traceback-file tb.txt --mock
  python scripts\\investigate.py --repo data\\bug_sample --traceback-file tb.txt --verify
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bugs.investigate import investigate  # noqa: E402
from core.bugs.render import investigation_to_text  # noqa: E402
from core.bugs.verify import verification_to_text, verify_investigation  # noqa: E402
from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.graph.expand import load_expansion_config, load_graph_context  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.sandbox.config import load_sandbox_config  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate a failure from a traceback")
    parser.add_argument("--repo", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traceback-file")
    source.add_argument("--traceback", help="the traceback text itself")
    parser.add_argument("--mock", action="store_true", help="use the MOCK provider")
    parser.add_argument("--no-judge", action="store_true", help="rule-based checks only")
    parser.add_argument("--verify", action="store_true", help="run the draft test in the sandbox")
    parser.add_argument("--sandbox-timeout", type=float, default=None, help="seconds per run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        llm = create_llm_from_env(provider_override="mock" if args.mock else None)
        qa_config = load_qa_config()
        validation = load_validation_config()
        expansion = load_expansion_config()
        sandbox = load_sandbox_config()
        if args.sandbox_timeout is not None:
            sandbox = dataclasses.replace(sandbox, timeout_seconds=args.sandbox_timeout)
        if args.traceback_file:
            text = Path(args.traceback_file).read_text(encoding="utf-8", errors="replace")
        else:
            text = args.traceback.replace("\\n", "\n")
    except (RepoLoadError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.no_judge:
        validation = dataclasses.replace(validation, judge_enabled=False)

    context = load_graph_context(root)
    print(f"Repository: {root} | symbols: {len(context.symbols)}")
    print(f"LLM: {llm.provider}/{llm.model_name}")
    inv = investigate(
        text, context, llm, qa_config=qa_config, validation_config=validation,
        expansion_config=expansion,
    )
    print("\n" + investigation_to_text(inv))
    if inv.hypothesis is None:
        return 3
    if not args.verify:
        return 0
    print("\nRunning the draft test in a disposable sandbox")
    print("(a temporary copy of the repository; your project folder is not touched)...")
    report = verify_investigation(inv, root, sandbox)
    print("\n" + verification_to_text(report))
    return 0 if report.verified else 4


if __name__ == "__main__":
    sys.exit(main())