# Local Semantic Search

A small local semantic search engine with:

- document loading for Markdown, text, and reStructuredText files
- chunking
- transformer embeddings through Sentence Transformers
- hashed word/character n-gram embeddings as a no-download fallback
- cosine similarity search
- persisted local JSON index
- CLI
- FastAPI web UI

By default, the indexer uses `sentence-transformers/all-MiniLM-L6-v2` when Sentence Transformers is installed. If that optional dependency is missing, it falls back to a dependency-light local hashing embedder.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For higher quality transformer embeddings:

```bash
pip install -e '.[transformers]'
```

## Build The Index

```bash
semantic-search build docs
```

Force transformer embeddings:

```bash
semantic-search build docs --backend transformer
```

Use the lightweight fallback explicitly:

```bash
semantic-search build docs --backend hashing
```

## Search From The CLI

```bash
semantic-search search "how do I answer questions over documents?"
```

## Run The Web App

```bash
uvicorn semantic_search.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Add Your Own Documents

Put `.md`, `.txt`, or `.rst` files in `docs/`, then run:

```bash
semantic-search build docs
```

or click `Reindex` in the web UI.
