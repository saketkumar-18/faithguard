"""Sparse (BM25) retrieval over chunks."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus = [tokenize(c.text + " " + c.title) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return list of (chunk_index, score), highest first.

        Falls back to raw token-overlap scoring when BM25 scores degenerate
        (e.g. tiny corpora where every IDF collapses to 0).
        """
        if not self._bm25:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = list(self._bm25.get_scores(q_tokens))
        if max(scores) <= 0:
            q_set = set(q_tokens)
            scores = [len(q_set & set(doc)) for doc in self._corpus]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
