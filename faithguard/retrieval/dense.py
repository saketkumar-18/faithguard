"""Dense retrieval with a sentence-transformer bi-encoder (CPU friendly)."""
from __future__ import annotations

import numpy as np

from .chunking import Chunk


class DenseIndex:
    """Embeds chunks once, then answers queries with cosine similarity.

    Uses sentence-transformers with the model pinned in config
    (default BAAI/bge-small-en-v1.5 — small, fast, strong on CPU).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        if not chunks:
            self._matrix = None
            return
        texts = [c.text for c in chunks]
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._matrix = emb.astype(np.float32)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._matrix is None:
            return []
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]
