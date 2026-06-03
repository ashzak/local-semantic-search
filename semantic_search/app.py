from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from semantic_search.documents import SUPPORTED_EXTENSIONS, read_document_text
from semantic_search.index import SearchIndex


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = Path(os.getenv("SEMANTIC_SEARCH_DOCS_PATH", ROOT / "docs"))
INDEX_PATH = Path(os.getenv("SEMANTIC_SEARCH_INDEX_PATH", ROOT / "data" / "index.json"))

app = FastAPI(title="Local Semantic Search")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")


@dataclass(frozen=True)
class HighlightedResult:
    score: float
    source: str
    title_html: str
    text_html: str


@dataclass(frozen=True)
class Answer:
    text_html: str
    sources: list[str]


@dataclass(frozen=True)
class DocumentView:
    filename: str
    title: str
    text: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "", notice: str = "", error: str = "") -> HTMLResponse:
    index = _load_or_build_index()
    raw_results = index.search(q, limit=8) if q.strip() else []
    results = [_highlight_result(result, q) for result in raw_results]
    answer = _build_answer(raw_results, q) if q.strip() else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": q,
            "results": results,
            "answer": answer,
            "chunk_count": len(index.chunks),
            "embedder": index.describe_embedder(),
            "documents": _list_documents(),
            "notice": notice,
            "error": error,
        },
    )


@app.get("/documents/{filename}", response_class=HTMLResponse)
def document_detail(request: Request, filename: str):
    path = _document_path(filename)
    if path is None or not path.exists():
        return _redirect_with_message(error="Document was not found.")

    document = DocumentView(
        filename=path.name,
        title=path.stem.replace("-", " ").replace("_", " ").title(),
        text=read_document_text(path).strip() or "No extractable text found.",
    )
    return templates.TemplateResponse(
        request,
        "document.html",
        {
            "document": document,
        },
    )


@app.post("/reindex")
def reindex() -> RedirectResponse:
    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return RedirectResponse("/", status_code=303)


@app.post("/search")
def search(q: str = Form(...)) -> RedirectResponse:
    return RedirectResponse(f"/?{urlencode({'q': q})}", status_code=303)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> RedirectResponse:
    filename = _safe_filename(file.filename or "")
    if not filename:
        return _redirect_with_message(error="Choose a document to upload.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return _redirect_with_message(error=f"Unsupported file type. Use {allowed}.")

    content = await file.read()
    if not content.strip():
        return _redirect_with_message(error="Uploaded document is empty.")

    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    destination = _available_path(DOCS_PATH / filename)
    destination.write_bytes(content)

    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return _redirect_with_message(notice=f"Uploaded and indexed {destination.name}.")


@app.post("/documents/delete")
def delete_document(filename: str = Form(...)) -> RedirectResponse:
    path = _document_path(filename)
    if path is None or not path.exists():
        return _redirect_with_message(error="Document was not found.")

    path.unlink()
    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return _redirect_with_message(notice=f"Deleted and reindexed {path.name}.")


def _load_or_build_index() -> SearchIndex:
    if INDEX_PATH.exists():
        return SearchIndex.load(INDEX_PATH)
    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return index


def _list_documents() -> list[str]:
    if not DOCS_PATH.exists():
        return []
    return sorted(
        path.name
        for path in DOCS_PATH.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        return ""
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    if not stem:
        stem = "document"
    return f"{stem}{suffix}"


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique filename for {path.name}")


def _document_path(filename: str) -> Path | None:
    name = Path(filename).name
    if name != filename or not name:
        return None

    path = DOCS_PATH / name
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    if not path.resolve().is_relative_to(DOCS_PATH.resolve()):
        return None
    return path


def _redirect_with_message(*, notice: str = "", error: str = "") -> RedirectResponse:
    params = {key: value for key, value in {"notice": notice, "error": error}.items() if value}
    return RedirectResponse(f"/?{urlencode(params)}", status_code=303)


def _highlight_result(result, query: str) -> HighlightedResult:
    terms = _highlight_terms(query)
    return HighlightedResult(
        score=result.score,
        source=result.chunk.source,
        title_html=_highlight_text(result.chunk.title, terms),
        text_html=_highlight_text(result.chunk.text, terms),
    )


def _highlight_terms(query: str) -> list[str]:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]{3,}", query)
        if not term.isdigit()
    }
    return sorted(terms, key=len, reverse=True)


def _highlight_text(text: str, terms: list[str]) -> str:
    escaped = html.escape(text)
    if not terms:
        return escaped

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


def _build_answer(results, query: str) -> Answer | None:
    terms = _highlight_terms(query)
    if not results:
        return None

    candidates: list[tuple[float, str, str]] = []
    for result in results[:4]:
        for sentence in _sentences(result.chunk.text):
            score = _sentence_score(sentence, terms)
            if score > 0:
                candidates.append((score + result.score, sentence, result.chunk.source))

    if not candidates:
        top = results[0]
        return Answer(
            text_html=_highlight_text(_compact_text(top.chunk.text), terms),
            sources=[top.chunk.source],
        )

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    selected = candidates[:3]
    answer_text = " ".join(sentence for _, sentence, _ in selected)
    sources = sorted({source for _, _, source in selected})
    return Answer(text_html=_highlight_text(answer_text, terms), sources=sources)


def _sentences(text: str) -> list[str]:
    compact = _compact_text(text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", compact)
        if len(sentence.strip()) >= 30
    ]


def _sentence_score(sentence: str, terms: list[str]) -> float:
    lowered = sentence.lower()
    return sum(1.0 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


def _compact_text(text: str) -> str:
    return " ".join(text.split())
