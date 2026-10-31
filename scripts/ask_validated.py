"""Ask a question and validate the answer from the command line.

Examples (from the project root):
  python scripts\\ask_validated.py --repo data\\sample_repo --question "How does checkout work?"
  python scripts\\ask_validated.py --repo data\\sample_repo --question "..." --mock --no-log
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
from core.chat.models import STATUS_ERROR, RepoInfo  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.validation.config import load_validation_config  # noqa: E402
from core.validation.engine import ValidatingQAEngine  # noqa: E402
from core.validation.render import validation_to_text  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question and validate the answer")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mock", action="store_true", help="use the MOCK provider")
    parser.add_argument("--no-judge", action="store_true", help="rule-based checks only")
    parser.add_argument("--no-log", action="store_true", help="do not write the evaluation log")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        llm = create_llm_from_env(provider_override="mock" if args.mock else None)
        qa_config = load_qa_config()
        config = load_validation_config()
    except (RepoLoadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.no_judge:
        config = dataclasses.replace(config, judge_enabled=False)
    if args.no_log:
        config = dataclasses.replace(config, log_enabled=False)

    embedder = SentenceTransformerEmbedder(embedding_model_name())
    engine = CodeSearchEngine(ScannedSymbolSource(root), embedder, default_db_path(root))
    try:
        stats = engine.sync()
        print(f"Indexed {stats.total} symbols. LLM: {llm.provider}/{llm.model_name}")
        repo = RepoInfo(root.name, stats.total)
        validated = ValidatingQAEngine(engine, llm, root, qa_config, config, repo).ask(
            args.question
        )
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.close()

    print("\n" + answer_to_markdown(validated.answer))
    print("\nValidation:\n" + validation_to_text(validated))
    if config.log_enabled:
        state = "written to" if validated.logged else "could NOT be written to"
        print(f"\nEvaluation record {state} {config.log_path}")
    return 3 if validated.answer.status == STATUS_ERROR else 0


if __name__ == "__main__":
    sys.exit(main())