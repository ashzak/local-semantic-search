from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

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


class HashingEmbedder:
    """Dependency-light embedding using word and character n-gram hashing."""

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
