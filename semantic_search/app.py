from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from semantic_search.index import SearchIndex


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "index.json"
DOCS_PATH = ROOT / "docs"

app = FastAPI(title="Local Semantic Search")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "") -> HTMLResponse:
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
        },
    )


@app.post("/reindex")
def reindex() -> RedirectResponse:
    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return RedirectResponse("/", status_code=303)


@app.post("/search")
def search(q: str = Form(...)) -> RedirectResponse:
    return RedirectResponse(f"/?q={q}", status_code=303)


def _load_or_build_index() -> SearchIndex:
    if INDEX_PATH.exists():
        return SearchIndex.load(INDEX_PATH)
    index = SearchIndex.build(DOCS_PATH)
    index.save(INDEX_PATH)
    return index
