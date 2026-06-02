from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
DEFAULT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER_CACHE: dict[tuple[str, int, str], "Embedder"] = {}

SYNONYMS = {
    "answer": ("question", "response", "qa"),
    "answers": ("question", "response", "qa"),
    "ask": ("question", "query", "qa"),
    "doc": ("document", "context", "source"),
    "docs": ("document", "context", "source"),
    "document": ("doc", "context", "source"),
    "documents": ("doc", "context", "source"),
    "find": ("search", "retrieve", "lookup"),
    "lookup": ("search", "retrieve", "find"),
    "meaning": ("semantic", "intent", "concept"),
    "question": ("answer", "query", "qa"),
    "questions": ("answer", "query", "qa"),
    "retrieve": ("search", "find", "lookup"),
    "retrieval": ("search", "context", "rag"),
    "search": ("retrieve", "find", "lookup"),
    "similar": ("similarity", "related", "matching"),
    "summarize": ("summary", "summarization", "brief"),
    "ticket": ("case", "issue", "support"),
}


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        ...

    def metadata(self) -> dict[str, str | int]:
        ...


class HashingEmbedder:
    """Dependency-light embedding using word and character n-gram hashing."""

    name = "hashing"

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        features = list(_features(text))
        if not features:
            return vector

        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def metadata(self) -> dict[str, str | int]:
        return {"backend": self.name, "dimensions": self.dimensions}


class SentenceTransformerEmbedder:
    """Transformer-backed embeddings from the sentence-transformers package."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_TRANSFORMER_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install -e '.[transformers]'"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            dimensions = self.model.get_embedding_dimension()
        else:
            dimensions = self.model.get_sentence_embedding_dimension()
        if dimensions is None:
            raise RuntimeError(f"Could not detect embedding dimensions for {model_name}")
        self.dimensions = int(dimensions)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.tolist() for embedding in embeddings]

    def metadata(self) -> dict[str, str | int]:
        return {
            "backend": self.name,
            "model": self.model_name,
            "dimensions": self.dimensions,
        }


def create_embedder(
    backend: str = "auto",
    *,
    dimensions: int = 2048,
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
) -> Embedder:
    resolved_backend = "sentence-transformers" if backend == "transformer" else backend
    if resolved_backend == "hashing":
        return _cached_embedder("hashing", dimensions, model_name)
    if resolved_backend == "sentence-transformers":
        return _cached_embedder("sentence-transformers", dimensions, model_name)
    if backend == "auto":
        try:
            return _cached_embedder("sentence-transformers", dimensions, model_name)
        except RuntimeError:
            return _cached_embedder("hashing", dimensions, model_name)
    raise ValueError(f"Unknown embedding backend: {backend}")


def clear_embedder_cache() -> None:
    _EMBEDDER_CACHE.clear()


def _cached_embedder(backend: str, dimensions: int, model_name: str) -> Embedder:
    key = (backend, dimensions, model_name)
    if key not in _EMBEDDER_CACHE:
        if backend == "hashing":
            _EMBEDDER_CACHE[key] = HashingEmbedder(dimensions=dimensions)
        elif backend == "sentence-transformers":
            _EMBEDDER_CACHE[key] = SentenceTransformerEmbedder(model_name=model_name)
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")
    return _EMBEDDER_CACHE[key]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _features(text: str) -> Iterable[tuple[str, float]]:
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)

    for token in tokens:
        yield f"w:{token}", 1.0
        for synonym in SYNONYMS.get(token, ()):
            yield f"w:{synonym}", 0.45

    for first, second in zip(tokens, tokens[1:], strict=False):
        yield f"b:{first}_{second}", 1.4

    compact = " ".join(tokens)
    for size, weight in ((3, 0.25), (4, 0.35), (5, 0.45)):
        for index in range(max(0, len(compact) - size + 1)):
            gram = compact[index : index + size]
            if " " not in gram:
                yield f"c{size}:{gram}", weight
