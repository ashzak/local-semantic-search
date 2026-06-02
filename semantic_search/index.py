from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from semantic_search.documents import Chunk, load_chunks
from semantic_search.embedding import HashingEmbedder, cosine


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: Chunk


class SearchIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        dimensions: int = 2048,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = HashingEmbedder(dimensions=dimensions)

    @classmethod
    def build(cls, documents_path: Path, dimensions: int = 2048) -> "SearchIndex":
        chunks = load_chunks(documents_path)
        embedder = HashingEmbedder(dimensions=dimensions)
        vectors = embedder.embed_many(chunk.text for chunk in chunks)
        return cls(chunks=chunks, vectors=vectors, dimensions=dimensions)

    @classmethod
    def load(cls, index_path: Path) -> "SearchIndex":
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        vectors = payload["vectors"]
        return cls(chunks=chunks, vectors=vectors, dimensions=payload["dimensions"])

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dimensions": self.embedder.dimensions,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": self.vectors,
        }
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_vector = self.embedder.embed(query)
        scored = [
            SearchResult(score=cosine(query_vector, vector), chunk=chunk)
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:limit]
