from pathlib import Path

from semantic_search.embedding import clear_embedder_cache, create_embedder
from semantic_search.index import SearchIndex


def test_hashing_index_round_trip_searches_documents(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "support.md").write_text(
        "# Support\n\nCustomers with billing issues need invoice help.",
        encoding="utf-8",
    )
    (docs / "search.md").write_text(
        "# Search\n\nSemantic search retrieves related documents.",
        encoding="utf-8",
    )

    index = SearchIndex.build(docs, backend="hashing")
    index_path = tmp_path / "index.json"
    index.save(index_path)

    loaded = SearchIndex.load(index_path)
    results = loaded.search("billing invoice customer", limit=1)

    assert loaded.describe_embedder() == "hashing"
    assert results[0].chunk.source == "support.md"


def test_create_embedder_reuses_cached_instances() -> None:
    clear_embedder_cache()

    first = create_embedder("hashing", dimensions=128)
    second = create_embedder("hashing", dimensions=128)
    third = create_embedder("hashing", dimensions=256)

    assert first is second
    assert first is not third


def test_loaded_indexes_reuse_cached_embedder(tmp_path: Path) -> None:
    clear_embedder_cache()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "support.md").write_text(
        "# Support\n\nCustomers with billing issues need invoice help.",
        encoding="utf-8",
    )

    index = SearchIndex.build(docs, backend="hashing")
    index_path = tmp_path / "index.json"
    index.save(index_path)

    first = SearchIndex.load(index_path)
    second = SearchIndex.load(index_path)

    assert first.embedder is second.embedder
