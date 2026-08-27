"""Dense retrieval with a sentence-embedding bi-encoder (CPU friendly).

Backend: fastembed (ONNX Runtime) — torch-free, ~100 MB RAM for
BAAI/bge-small-en-v1.5. Same model family as training, so the baked
embedding cache stays valid.
"""
from __future__ import annotations

import numpy as np

from .chunking import Chunk


class DenseIndex:
    """Embeds chunks once, then answers queries with cosine similarity.

    Uses fastembed with the model pinned in config
    (default BAAI/bge-small-en-v1.5 — small, fast, strong on CPU).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # fastembed returns L2-normalized embeddings by default
        emb = np.asarray(
            list(self.model.embed(texts, batch_size=batch_size)), dtype=np.float32
        )
        # normalize defensively in case the backend changes defaults
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        if not chunks:
            self._matrix = None
            return
        texts = [c.text for c in chunks]
        self._matrix = self._encode(texts, batch_size=batch_size)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._matrix is None:
            return []
        q = self._encode([query])[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]

    # ------------------------------------------------------------- caching
    def save_cache(self, path) -> None:
        """Persist the embedding matrix so rebuilds don't re-encode."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.savez_compressed(path, matrix=self._matrix)

    def load_cache(self, path) -> bool:
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return False
        data = np.load(path)
        self._matrix = data["matrix"].astype(np.float32)
        return True
