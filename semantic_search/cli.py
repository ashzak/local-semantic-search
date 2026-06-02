from __future__ import annotations

import argparse
from pathlib import Path

from semantic_search.embedding import DEFAULT_TRANSFORMER_MODEL
from semantic_search.index import SearchIndex


DEFAULT_INDEX = Path("data/index.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local semantic search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a search index")
    build_parser.add_argument("documents", type=Path, help="File or directory to index")
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build_parser.add_argument(
        "--backend",
        choices=("auto", "transformer", "hashing"),
        default="auto",
        help="Embedding backend. Auto uses transformers when installed.",
    )
    build_parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL)
    build_parser.add_argument("--dimensions", type=int, default=2048)

    search_parser = subparsers.add_parser("search", help="Search an index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        index = SearchIndex.build(
            args.documents,
            backend=args.backend,
            dimensions=args.dimensions,
            model_name=args.model,
        )
        index.save(args.index)
        print(
            f"Indexed {len(index.chunks)} chunks into {args.index} "
            f"using {index.describe_embedder()}"
        )
        return

    if args.command == "search":
        index = SearchIndex.load(args.index)
        print(f"Using {index.describe_embedder()}")
        print()
        for result in index.search(args.query, limit=args.limit):
            print(f"{result.score:.3f}  {result.chunk.source}  {result.chunk.title}")
            print(_excerpt(result.chunk.text))
            print()


def _excerpt(text: str, length: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return f"{compact[: length - 3]}..."
