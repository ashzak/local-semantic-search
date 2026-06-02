from __future__ import annotations

import argparse
from pathlib import Path

from semantic_search.index import SearchIndex


DEFAULT_INDEX = Path("data/index.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local semantic search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a search index")
    build_parser.add_argument("documents", type=Path, help="File or directory to index")
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build_parser.add_argument("--dimensions", type=int, default=2048)

    search_parser = subparsers.add_parser("search", help="Search an index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        index = SearchIndex.build(args.documents, dimensions=args.dimensions)
        index.save(args.index)
        print(f"Indexed {len(index.chunks)} chunks into {args.index}")
        return

    if args.command == "search":
        index = SearchIndex.load(args.index)
        for result in index.search(args.query, limit=args.limit):
            print(f"{result.score:.3f}  {result.chunk.source}  {result.chunk.title}")
            print(_excerpt(result.chunk.text))
            print()


def _excerpt(text: str, length: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return f"{compact[: length - 3]}..."
