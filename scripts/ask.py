"""Ask a question about a repository from the command line.

Examples (from the project root):
  python scripts\\ask.py --repo data\\sample_repo --question "How does checkout work?" --mock
  python scripts\\ask.py --repo data\\sample_repo --question "Where is login handled?" --scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chat.config import create_llm_from_env, load_qa_config  # noqa: E402
from core.chat.models import STATUS_ERROR, RepoInfo  # noqa: E402
from core.chat.qa import QAEngine, is_relevant  # noqa: E402
from core.chat.render import answer_to_markdown  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question about a repository")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mock", action="store_true", help="use the MOCK provider (no model)")
    parser.add_argument("--scores", action="store_true", help="show retrieval scores (tuning)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        llm = create_llm_from_env(provider_override="mock" if args.mock else None)
        config = load_qa_config()
    except (RepoLoadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    embedder = SentenceTransformerEmbedder(embedding_model_name())
    engine = CodeSearchEngine(ScannedSymbolSource(root), embedder, default_db_path(root))
    try:
        stats = engine.sync()
        print(f"Indexed {stats.total} symbols. LLM: {llm.provider}/{llm.model_name}")
        if args.scores:
            print("\nRetrieval (relevant = passes the evidence gate):")
            for rank, r in enumerate(engine.search(args.question, "hybrid", config.candidate_k), 1):
                cos = f"{r.semantic_score:.2f}" if r.semantic_score is not None else "-"
                bm25 = f"{r.keyword_score:.1f}" if r.keyword_score is not None else "-"
                flag = "relevant" if is_relevant(r, config) else "ignored"
                print(f"{rank:>2}. {r.symbol.qualname} {r.symbol.location} "
                      f"cos={cos} bm25={bm25} [{flag}]")
        qa = QAEngine(engine, llm, config, RepoInfo(root.name, stats.total))
        answer = qa.ask_safe(args.question)
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.close()

    print("\n" + answer_to_markdown(answer))
    for warning in answer.warnings:
        print(f"\nWARNING: {warning}")
    return 3 if answer.status == STATUS_ERROR else 0


if __name__ == "__main__":
    sys.exit(main())