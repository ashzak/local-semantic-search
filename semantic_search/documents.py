from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> "Chunk":
        return cls(
            id=value["id"],
            source=value["source"],
            title=value["title"],
            text=value["text"],
        )


def load_chunks(path: Path, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")

    files = _iter_files(path)
    chunks: list[Chunk] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        title = _title_for(file_path, text)
        for index, piece in enumerate(_split_text(text, chunk_size, overlap), start=1):
            relative = str(file_path.relative_to(path)) if path.is_dir() else file_path.name
            chunks.append(
                Chunk(
                    id=f"{relative}#{index}",
                    source=relative,
                    title=title,
                    text=piece,
                )
            )

    return chunks


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or path.stem
    return path.stem.replace("-", " ").replace("_", " ").title()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.extend(_split_long_piece(current, chunk_size, overlap))
            current = paragraph

    if current:
        chunks.extend(_split_long_piece(current, chunk_size, overlap))

    return chunks


def _split_long_piece(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [piece for piece in pieces if piece]
