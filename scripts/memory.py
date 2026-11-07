"""Commit-aware repository memory from the command line.

Examples (from the project root):
  python scripts\\memory.py status --repo .
  python scripts\\memory.py update --repo . [--no-embeddings] [--full]
  python scripts\\memory.py ask --repo . --question "How does checkout work?" [--policy warn]
  python scripts\\memory.py history --repo . [--show 3]

Exit codes: status 0 = fresh, 3 = behind, stale or not indexed. ask 0 = answered,
3 = blocked, 2 = problem with the embedding model, 1 = bad input.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.memory.answers import AnswerStore  # noqa: E402
from core.memory.config import STALE_POLICIES, load_memory_config  # noqa: E402
from core.memory.freshness import check_freshness  # noqa: E402
from core.memory.render import (  # noqa: E402
    citation_rows,
    freshness_lines,
    history_lines,
    record_markdown,
    stamp_text,
    versioned_text,
)
from core.memory.service import MemoryService  # noqa: E402
from core.memory.store import MemoryStore, memory_db_path  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.semantic.config import embedding_model_name  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402


def make_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(embedding_model_name())


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit-aware repository memory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "update", "ask", "history"):
        command = sub.add_parser(name)
        command.add_argument("--repo", required=True)
        if name == "update":
            command.add_argument("--no-embeddings", action="store_true")
            command.add_argument("--full", action="store_true", help="recompute the whole graph")
        if name == "ask":
            command.add_argument("--question", required=True)
            command.add_argument("--mock", action="store_true")
            command.add_argument("--no-judge", action="store_true")
            command.add_argument("--policy", choices=STALE_POLICIES, default=None)
            command.add_argument("--reuse", action="store_true")
        if name == "history":
            command.add_argument("--limit", type=int, default=20)
            command.add_argument("--show", type=int, default=None)
    return parser.parse_args(argv)


def cmd_status(root: Path, args: argparse.Namespace) -> int:
    store = MemoryStore(memory_db_path(root))
    try:
        fresh = check_freshness(root, store)
    finally:
        store.close()
    print("\n".join(freshness_lines(fresh)))
    return 0 if fresh.is_fresh else 3


def cmd_update(root: Path, args: argparse.Namespace) -> int:
    embedder = None if args.no_embeddings else make_embedder()
    service = MemoryService(root, embedder=embedder, config=load_memory_config())
    try:
        update = service.refresh(force_full=args.full, note="update from the command line")
    finally:
        service.close()
    print(update.describe())
    return 0


def cmd_ask(root: Path, args: argparse.Namespace) -> int:
    llm = create_llm_from_env(provider_override="mock" if args.mock else None)
    validation = load_validation_config()
    if args.no_judge:
        validation = dataclasses.replace(validation, judge_enabled=False)
    service = MemoryService(root, embedder=make_embedder(), config=load_memory_config())
    try:
        answer = service.ask(
            args.question, llm, qa_config=load_qa_config(), validation_config=validation,
            policy=args.policy, reuse=args.reuse,
        )
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        service.close()
    print(versioned_text(answer))
    return 3 if answer.blocked else 0


def cmd_history(root: Path, args: argparse.Namespace) -> int:
    store = MemoryStore(memory_db_path(root))
    try:
        answers = AnswerStore(store)
        if args.show is None:
            print("\n".join(history_lines(answers.recent(args.limit))))
            return 0
        record = answers.get(args.show)
        if record is None:
            print(f"ERROR: there is no stored answer #{args.show}", file=sys.stderr)
            return 1
        print(f"Answer #{record.id}: {record.question}")
        print(f"Based on {stamp_text(record.version)}")
        if record.version.commit_hash:
            print(f"Commit: {record.version.commit_hash}")
        print(f"Index was {record.freshness} when asked\n")
        print(record_markdown(record))
        for row in citation_rows(record.citations):
            print(f"  {row['citation']} {row['where']} @ {row['version']} valid: {row['valid']}")
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        handler = {
            "status": cmd_status, "update": cmd_update, "ask": cmd_ask, "history": cmd_history,
        }[args.command]
        return handler(root, args)
    except (RepoLoadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())