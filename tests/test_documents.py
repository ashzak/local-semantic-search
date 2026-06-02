from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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


def test_load_chunks_extracts_pdf_text(tmp_path: Path) -> None:
    pdf = tmp_path / "billing-playbook.pdf"
    _write_text_pdf(pdf, "PDF billing refund guidance")

    chunks = load_chunks(pdf)

    assert len(chunks) == 1
    assert chunks[0].source == "billing-playbook.pdf"
    assert chunks[0].title == "Billing Playbook"
    assert "billing refund" in chunks[0].text


def test_load_chunks_skips_malformed_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")

    assert load_chunks(pdf) == []


def _write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 120 Td ({text}) Tj ET".encode("utf-8"))
    page.replace_contents(stream)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )

    with path.open("wb") as output:
        writer.write(output)
