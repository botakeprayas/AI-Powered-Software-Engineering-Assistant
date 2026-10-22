"""Scan a repository from the command line (no UI).

Examples (from the project root):
  python scripts\\scan_repo.py data\\sample_repo
  python scripts\\scan_repo.py data\\sample_repo --search validate_email
  python scripts\\scan_repo.py . --skipped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.pipeline import analyze_repository  # noqa: E402
from core.settings import load_settings  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a repository and search its symbols")
    parser.add_argument("folder", help="repository folder")
    parser.add_argument("--search", default=None, help="keyword(s) to search for")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tree", action="store_true", help="print the file tree")
    parser.add_argument("--skipped", action="store_true", help="print skipped files and reasons")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.folder)
    except RepoLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    analysis = analyze_repository(root)
    try:
        print(f"Scanned: {analysis.root}")
        print(
            f"Python files: {len(analysis.python_files)} | symbols: {len(analysis.symbols)} | "
            f"skipped: {len(analysis.skipped)} | warnings: {len(analysis.issues)}"
        )
        if args.tree:
            print("\n" + analysis.tree_text())
        if args.skipped:
            print("\nSkipped:")
            for item in analysis.skipped:
                print(f"  [{item.kind}] {item.path}: {item.reason}")
        if analysis.issues:
            print("\nWarnings:")
            for issue in analysis.issues:
                print(f"  {issue.path}: {issue.message}")
        if args.search:
            hits = analysis.index.search(args.search, limit=args.limit)
            print(f"\nSearch {args.search!r}: {len(hits)} result(s)")
            for number, hit in enumerate(hits, start=1):
                sym = hit.symbol
                print(f"{number:>2}. {sym.qualname} [{sym.kind}] {sym.location}")
                print(f"    score {hit.score} | matched in: {', '.join(hit.matched)}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        analysis.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())