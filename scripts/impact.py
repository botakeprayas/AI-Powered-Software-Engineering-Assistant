"""Change-impact analysis from the command line.

Examples (from the project root):
  python scripts\\impact.py --repo data\\impact_sample --diff-file change.diff
  git diff | python scripts\\impact.py --repo . --diff-file -
  python scripts\\impact.py --repo . --diff-file c.diff --expect-files "a.py,b.py"

Scores are heuristic ranks, not probabilities. Exit codes: 0 report, 1 bad input,
3 nothing in the diff could be mapped to indexed Python code.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.graph.expand import load_graph_context  # noqa: E402
from core.impact.analyze import (  # noqa: E402
    analyze_impact,
    compare_files,
    compare_tests,
    default_signals,
    load_impact_config,
)
from core.impact.changes import IndexView, map_diff  # noqa: E402
from core.impact.diff_parser import parse_diff  # noqa: E402
from core.impact.render import report_to_text  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.settings import load_settings  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Change-impact analysis for a diff")
    parser.add_argument("--repo", required=True, help="the repository folder (indexed)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--diff-file", help="a diff file, or - to read standard input")
    source.add_argument("--diff", help="the diff text itself")
    parser.add_argument(
        "--base", choices=("auto", "old", "new"), default="auto",
        help="which version the folder holds: before the change, with it, or detect",
    )
    parser.add_argument("--no-git", action="store_true", help="do not use git history")
    parser.add_argument("--no-ruff", action="store_true", help="do not run Ruff")
    parser.add_argument("--depth", type=int, default=None, help="trace depth (1-5)")
    parser.add_argument("--expect-files", default=None, help="comma-separated hand-checked files")
    parser.add_argument("--expect-tests", default=None, help="comma-separated hand-checked tests")
    return parser.parse_args(argv)


def split_list(text: str | None) -> list[str]:
    if not text:
        return []
    items = (part.strip().replace("\\", "/") for part in text.split(","))
    return [item for item in items if item]


def read_diff(args: argparse.Namespace) -> str:
    if args.diff is not None:
        return args.diff.replace("\\n", "\n")
    if args.diff_file == "-":
        return sys.stdin.read()
    return Path(args.diff_file).read_text(encoding="utf-8-sig", errors="replace")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        config = load_impact_config()
        config = dataclasses.replace(
            config,
            use_git=config.use_git and not args.no_git,
            use_ruff=config.use_ruff and not args.no_ruff,
        )
        if args.depth is not None:
            traversal = dataclasses.replace(config.traversal, max_depth=args.depth)
            config = dataclasses.replace(config, traversal=traversal)
        text = read_diff(args)
    except (RepoLoadError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    context = load_graph_context(root)
    view = IndexView.from_context(context)
    changes = map_diff(parse_diff(text), view, args.base)
    report = analyze_impact(changes, context, view, config, default_signals(root, config))
    comparisons = {}
    files = split_list(args.expect_files)
    tests = split_list(args.expect_tests)
    if files:
        comparisons["affected files"] = compare_files(report, files)
    if tests:
        comparisons["tests"] = compare_tests(report, tests)
    print(report_to_text(report, comparisons))
    return 0 if report.units else 3


if __name__ == "__main__":
    sys.exit(main())