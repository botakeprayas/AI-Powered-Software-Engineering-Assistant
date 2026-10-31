"""Ask a question with call-graph expansion from the command line.

Examples (from the project root):
    python scripts\\ask_graph.py --repo data\\graph_sample --question "Who calls apply_tax?" --mock
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
from core.graph.expand import GraphEvidence, load_expansion_config, load_graph_context  # noqa: E402
from core.graph.qa import GraphQAEngine  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.repo.loader import RepoLoadError, validate_folder  # noqa: E402
from core.repo.source import ScannedSymbolSource  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import CodeSearchEngine  # noqa: E402
from core.settings import load_settings  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask with call-graph expansion")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mock", action="store_true", help="use the MOCK provider")
    parser.add_argument("--depth", type=int, default=None, help="call-graph depth (1-3)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    try:
        root = validate_folder(args.repo)
        llm = create_llm_from_env(provider_override="mock" if args.mock else None)
        qa_config = load_qa_config()
        expansion = load_expansion_config()
        if args.depth is not None:
            expansion = dataclasses.replace(expansion, depth=args.depth)
    except (RepoLoadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    embedder = SentenceTransformerEmbedder(embedding_model_name())
    engine = CodeSearchEngine(ScannedSymbolSource(root), embedder, default_db_path(root))
    try:
        stats = engine.sync()
        context = load_graph_context(root)
        print(f"Indexed {stats.total} symbols. LLM: {llm.provider}/{llm.model_name}")
        qa = GraphQAEngine(
            engine, llm, context, qa_config, RepoInfo(root.name, stats.total), expansion
        )
        answer = qa.ask_safe(args.question)
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.close()

    print("\n" + answer_to_markdown(answer))
    added = [e for e in answer.evidence if isinstance(e, GraphEvidence)]
    print(f"\nCall-graph context: {len(added)} excerpt(s) added")
    for item in added:
        print(f"  {item.id} {item.symbol.qualname} {item.location} <- {item.via}")
    for note in qa.last_expansion.notes:
        print(f"  note: {note}")
    for reason in qa.last_expansion.skipped:
        print(f"  not added: {reason}")
    return 3 if answer.status == STATUS_ERROR else 0


if __name__ == "__main__":
    sys.exit(main())