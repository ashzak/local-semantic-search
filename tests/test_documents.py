from pathlib import Path

import pytest

from semantic_search.documents import load_chunks


def test_load_chunks_extracts_titles_and_ignores_unsupported_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text("# Billing Guide\n\nInvoices and refunds.", encoding="utf-8")
    (docs / "ignored.py").write_text("print('skip')", encoding="utf-8")

    chunks = load_chunks(docs)

    assert len(chunks) == 1
    assert chunks[0].source == "billing.md"
    assert chunks[0].title == "Billing Guide"


def test_load_chunks_rejects_invalid_chunk_settings(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Test\n\nBody", encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        load_chunks(doc, chunk_size=100, overlap=100)
