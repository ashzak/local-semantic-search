from pathlib import Path

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
