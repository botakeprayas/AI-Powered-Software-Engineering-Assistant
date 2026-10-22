"""Search a repository by meaning, keyword, or both.

Examples (from the project root):
  python scripts\\semantic_search.py --repo . --query "load configuration from environment"
  python scripts\\semantic_search.py --repo . --query load_settings --mode keyword
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_config import setup_logging  # noqa: E402
from core.semantic.config import default_db_path, embedding_model_name  # noqa: E402
from core.semantic.engine import MODES, CodeSearchEngine  # noqa: E402
from core.semantic.symbol_source import AstSymbolSource  # noqa: E402
from core.settings import load_settings  # noqa: E402
from services.embedding_service import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid code search")
    parser.add_argument("--repo", default=".", help="repository folder to index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=None, help="override EMBEDDING_MODEL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(load_settings().log_level)
    repo = Path(args.repo)
    source = AstSymbolSource(repo)
    embedder = SentenceTransformerEmbedder(args.model or embedding_model_name())
    engine = CodeSearchEngine(source, embedder, default_db_path(repo))
    try:
        stats = engine.sync(semantic=args.mode != "keyword")
        if args.mode == "keyword":
            print(f"Loaded {stats.total} symbols (keyword mode: no embeddings used)")
        else:
            print(
                f"Indexed {stats.total} symbols: {stats.embedded} embedded, "
                f"{stats.reused} reused, {stats.removed} removed ({stats.seconds:.1f}s)"
            )
        for problem in source.errors:
            print(f"  skipped {problem}")
        results = engine.search(args.query, mode=args.mode, top_k=args.top_k)
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Tip: --mode keyword works without the embedding model.", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.close()

    print(f"\nQuery: {args.query!r} ({args.mode}, top {args.top_k})\n")
    if not results:
        print("No results.")
        return 0
    label = {"semantic": "cosine", "keyword": "BM25", "hybrid": "RRF"}[args.mode]
    for i, r in enumerate(results, start=1):
        sym = r.symbol
        print(f"{i:>2}. {sym.qualname}  [{sym.kind}]  {sym.location}")
        parts = [f"score {r.score:.4f} ({label})"]
        if r.semantic_rank is not None:
            parts.append(f"semantic #{r.semantic_rank} cos={r.semantic_score:.3f}")
        if r.keyword_rank is not None:
            parts.append(f"keyword #{r.keyword_rank} bm25={r.keyword_score:.2f}")
        print("    " + " | ".join(parts))
        if sym.docstring:
            print(f"    {sym.docstring.splitlines()[0][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())