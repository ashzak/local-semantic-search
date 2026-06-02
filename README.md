# Local Semantic Search

A small local semantic search engine with:

- document loading for Markdown, text, and reStructuredText files
- chunking
- hashed word/character n-gram embeddings
- cosine similarity search
- persisted local JSON index
- CLI
- FastAPI web UI

The default embedder is dependency-light so the project runs locally without downloading a model. For stronger semantic quality, replace `HashingEmbedder` with a transformer-backed embedder such as a Sentence Transformers model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Build The Index

```bash
semantic-search build docs
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
