from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from semantic_search.documents import SUPPORTED_EXTENSIONS
from semantic_search.index import SearchIndex


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "index.json"
DOCS_PATH = ROOT / "docs"

app = FastAPI(title="Local Semantic Search")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "", notice: str = "", error: str = "") -> HTMLResponse:
    index = _load_or_build_index()
    results = index.search(q, limit=8) if q.strip() else []
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": q,
            "results": results,
            "chunk_count": len(index.chunks),
            "embedder": index.describe_embedder(),
            "documents": _list_documents(),
            "notice": notice,
            "error": error,
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


def _redirect_with_message(*, notice: str = "", error: str = "") -> RedirectResponse:
    params = {key: value for key, value in {"notice": notice, "error": error}.items() if value}
    return RedirectResponse(f"/?{urlencode(params)}", status_code=303)
