from pathlib import Path

from fastapi.testclient import TestClient

import semantic_search.app as app_module
from semantic_search.app import _build_answer, _document_path, _highlight_text
from semantic_search.documents import Chunk
from semantic_search.index import SearchResult


def test_highlight_text_escapes_html_before_marking() -> None:
    result = _highlight_text("<script>billing issue</script>", ["billing", "issue"])

    assert "&lt;script&gt;" in result
    assert "<mark>billing</mark>" in result
    assert "<script>" not in result


def test_build_answer_selects_matching_sentences() -> None:
    result = SearchResult(
        score=0.5,
        chunk=Chunk(
            id="support.md#1",
            source="support.md",
            title="Support",
            text=(
                "Support teams route billing issues to operations. "
                "Unrelated product news should not matter."
            ),
        ),
    )

    answer = _build_answer([result], "billing operations")

    assert answer is not None
    assert answer.sources == ["support.md"]
    assert "<mark>billing</mark>" in answer.text_html


def test_document_path_rejects_traversal(monkeypatch, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(app_module, "DOCS_PATH", docs)

    assert _document_path("safe.md") == docs / "safe.md"
    assert _document_path("../README.md") is None
    assert _document_path("bad.py") is None


def test_upload_and_delete_document_routes(monkeypatch, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    data = tmp_path / "data"
    docs.mkdir()
    data.mkdir()
    (docs / "base.md").write_text(
        "# Base\n\nSemantic search has a starting document.",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "DOCS_PATH", docs)
    monkeypatch.setattr(app_module, "INDEX_PATH", data / "index.json")

    client = TestClient(app_module.app)
    upload_response = client.post(
        "/upload",
        files={"file": ("billing notes.md", b"# Billing\n\nInvoice refunds and charges.", "text/markdown")},
        follow_redirects=True,
    )

    assert upload_response.status_code == 200
    assert "Uploaded and indexed billing-notes.md." in upload_response.text
    assert (docs / "billing-notes.md").exists()

    delete_response = client.post(
        "/documents/delete",
        data={"filename": "billing-notes.md"},
        follow_redirects=True,
    )

    assert delete_response.status_code == 200
    assert "Deleted and reindexed billing-notes.md." in delete_response.text
    assert not (docs / "billing-notes.md").exists()


def test_upload_rejects_unsupported_file(monkeypatch, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    data = tmp_path / "data"
    docs.mkdir()
    data.mkdir()
    (docs / "base.md").write_text("# Base\n\nContent.", encoding="utf-8")
    monkeypatch.setattr(app_module, "DOCS_PATH", docs)
    monkeypatch.setattr(app_module, "INDEX_PATH", data / "index.json")

    client = TestClient(app_module.app)
    response = client.post(
        "/upload",
        files={"file": ("script.py", b"print('no')", "text/x-python")},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Unsupported file type." in response.text
    assert not (docs / "script.py").exists()
