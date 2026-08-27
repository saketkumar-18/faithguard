"""Document chunking with overlap and metadata preservation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class Chunker:
    """Split documents into overlapping chunks, preferring sentence boundaries.

    Parameters
    ----------
    chunk_size: target chunk length in characters.
    chunk_overlap: characters of overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str, title: str = "", meta: dict | None = None) -> list[Chunk]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        meta = meta or {}
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                # try to end on a sentence boundary inside the window
                window = text[start:end]
                boundaries = [m.end() for m in _SENT_BOUNDARY.finditer(window)]
                # keep at least 60% of the window so we don't emit tiny chunks
                good = [b for b in boundaries if b >= 0.6 * self.chunk_size]
                if good:
                    end = start + good[-1]
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(doc_id=doc_id, chunk_id=idx, text=piece, title=title, meta=dict(meta)))
                idx += 1
            if end >= n:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[Chunk]:
        """documents: [{"id": str, "text": str, "title": str?, "meta": dict?}]"""
        out: list[Chunk] = []
        for doc in documents:
            out.extend(
                self.chunk_text(
                    doc["text"],
                    doc_id=str(doc["id"]),
                    title=doc.get("title", ""),
                    meta=doc.get("meta"),
                )
            )
        return out
