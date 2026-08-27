"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index
from .chunking import Chunk
from .dense import DenseIndex


@dataclass
class RetrievedPassage:
    text: str
    doc_id: str
    chunk_id: int
    score: float          # fused RRF score
    bm25_rank: int | None
    dense_rank: int | None
    title: str = ""

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


class HybridRetriever:
    """BM25 + dense bi-encoder fused via weighted Reciprocal Rank Fusion.

    RRF score for document d:  sum over rankers r of  w_r / (k + rank_r(d))
    with k=60 (Cormack et al., 2009). Robust to the very different score
    scales of BM25 and cosine similarity — no per-query normalization needed.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        use_dense: bool = True,
    ):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Index(chunks)
        self.dense: DenseIndex | None = None
        if use_dense and chunks:
            self.dense = DenseIndex(embedding_model, device=device)
            self.dense.index(chunks)

    def search(self, query: str, top_k: int = 5, candidate_k: int = 50) -> list[RetrievedPassage]:
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        dense_hits = self.dense.search(query, top_k=candidate_k) if self.dense else []

        scores: dict[int, float] = {}
        bm25_rank: dict[int, int] = {}
        dense_rank: dict[int, int] = {}

        for rank, (idx, _s) in enumerate(bm25_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + self.bm25_weight / (self.rrf_k + rank)
            bm25_rank[idx] = rank
        for rank, (idx, _s) in enumerate(dense_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + self.dense_weight / (self.rrf_k + rank)
            dense_rank[idx] = rank

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievedPassage] = []
        for idx, score in ordered:
            c = self.chunks[idx]
            out.append(
                RetrievedPassage(
                    text=c.text,
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    score=score,
                    bm25_rank=bm25_rank.get(idx),
                    dense_rank=dense_rank.get(idx),
                    title=c.title,
                )
            )
        return out
