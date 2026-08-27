"""FastAPI app: guarded RAG over a loaded corpus, plus detection-only scoring.

Endpoints
---------
GET  /health                 liveness + model status (no auth)
GET  /metrics                observability counters (no auth)
GET  /corpus                 corpus stats
POST /ask                    full guarded pipeline (retrieve+generate+detect+mitigate)
POST /detect                 score a pre-generated answer against given passages
POST /corpus/load            load a new corpus at runtime (list of documents)

Production hardening
--------------------
- API-key auth via ``FG_API_KEY`` (x-api-key header or Authorization: Bearer ***
  Disabled when unset (dev), with a loud startup warning.
- Rate limiting via ``FG_RATE_LIMIT`` (requests per ``FG_RATE_WINDOW_S``).
- Request IDs + structured access logs + /metrics.
- Circuit breaker on LLM calls (fails fast after repeated 5xx).
- Graceful shutdown on SIGTERM/SIGINT.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from faithguard.api.observability import metrics
from faithguard.api.security import configured_api_key, auth_enabled, _extract_key, require_api_key, build_rate_limiter, enforce_rate_limit
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..circuit_breaker import CircuitBreaker
from ..config import DATA_DIR, MODELS_DIR, get_settings
from ..detection.classifier import HallucinationClassifier
from ..detection.nli import NLIScorer
from ..generation.llm import LLMClient
from ..pipeline import GuardedRAGPipeline
from ..retrieval.chunking import Chunker
from ..retrieval.hybrid import HybridRetriever
from .observability import AccessLogMiddleware, RequestIdMiddleware, metrics
from .security import (
    auth_enabled,
    build_rate_limiter,
    enforce_rate_limit,
    require_api_key,
)

log = logging.getLogger("faithguard.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mitigate: bool = True


class DetectRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    passages: list[str] = Field(min_length=1, max_length=20)


class LoadCorpusRequest(BaseModel):
    documents: list[dict] = Field(min_length=1, max_length=5000)
    # each: {"id": str, "text": str, "title": str?}


class AppState:
    def __init__(self):
        self.settings = get_settings()
        self.nli: NLIScorer | None = None
        self.classifier: HallucinationClassifier | None = None
        self.llm: LLMClient | None = None
        self._corpus_path = None  # deferred corpus load (512 MB free tier)
        self.pipeline: GuardedRAGPipeline | None = None
        self.n_chunks = 0
        self.n_docs = 0
        self.loaded_at: float | None = None


def _build_pipeline(state: AppState, documents: list[dict]) -> None:
    s = state.settings
    chunker = Chunker(s.retrieval.chunk_size, s.retrieval.chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    retriever = HybridRetriever(
        chunks,
        embedding_model=s.retrieval.embedding_model,
        device=s.device,
        rrf_k=s.retrieval.rrf_k,
        bm25_weight=s.retrieval.bm25_weight,
        dense_weight=s.retrieval.dense_weight,
        use_dense=False,  # attach dense index below with caching
    )
    if s.retrieval.use_dense:
        from ..retrieval.dense import DenseIndex

        dense = DenseIndex(s.retrieval.embedding_model, device=s.device)
        cache = DATA_DIR / "embed_cache.npz"
        if dense.load_cache(cache):
            dense.chunks = chunks
            log.info("Loaded embedding cache from %s", cache.name)
        else:
            log.info("Embedding %d chunks (cached afterwards) ...", len(chunks))
            dense.index(chunks)
            try:
                dense.save_cache(cache)
            except OSError as e:  # read-only data dir is fine
                log.warning("Could not write embedding cache: %s", e)
        retriever.dense = dense
    else:
        log.info("FG_USE_DENSE=0 — BM25-only retrieval (dense model not loaded)")
    state.pipeline = GuardedRAGPipeline(retriever, state.llm, state.nli, state.classifier, s)
    state.n_chunks = len(chunks)
    state.n_docs = len(documents)
    state.loaded_at = time.time()


def create_app(load_default_corpus: bool = True) -> FastAPI:
    state = AppState()
    limiter = build_rate_limiter()
    # observability singleton is already imported: metrics

    # ... rest of app setup (lifespan, endpoints, etc.)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.time()
        if not auth_enabled():
            log.warning(
                "FG_API_KEY is not set — authentication is DISABLED. "
                "Set FG_API_KEY before exposing this service publicly."
            )
        if limiter is None:
            log.warning("FG_RATE_LIMIT is not set — rate limiting is DISABLED.")
        log.info("Loading NLI model %s ...", state.settings.detection.nli_model)
        state.nli = NLIScorer(state.settings.detection.nli_model, device=state.settings.device)
        state.classifier = HallucinationClassifier(
            MODELS_DIR / "hallucination_classifier.pkl",
            unsupported_threshold=state.settings.detection.unsupported_claim_threshold,
            hallucinated_fraction=state.settings.detection.hallucinated_answer_fraction,
        )
        state.llm = LLMClient.from_settings(state.settings)
        # Defer corpus loading to first /ask: loading NLI + corpus together
        # at startup peaks past 512 MB on the Render free tier. Startup now
        # loads only NLI + classifier (~435 MB); the corpus is built lazily.
        if load_default_corpus:
            corpus_path = DATA_DIR / "corpus.json"
            if corpus_path.exists():
                state._corpus_path = corpus_path
                log.info("Corpus deferred to first /ask (%s)", corpus_path.name)
            else:
                log.warning("No corpus.json found; call POST /corpus/load before /ask")
        log.info("FaithGuard API ready in %.1fs", time.time() - t0)
        yield
        # graceful shutdown: uvicorn's --timeout-graceful-shutdown lets
        # in-flight requests finish before the process exits.
        log.info("Shutting down; in-flight requests will drain.")

    app = FastAPI(
        title="FaithGuard",
        description="Hallucination Detection & Mitigation Engine for RAG",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.fg = state

    # observability middleware (order matters: outermost first)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)

    def _guard(request: Request):
        """Combined auth + rate-limit dependency for protected endpoints."""
        require_api_key(request)
        enforce_rate_limit(limiter, request)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "nli_loaded": state.nli is not None,
            "classifier_loaded": state.classifier is not None,
            "classifier_method": (
                "trained" if state.classifier and state.classifier.model else "rules"
            ),
            "llm_configured": bool(state.llm and state.llm.api_key),
            "corpus": {"docs": state.n_docs, "chunks": state.n_chunks},
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint():
        return metrics.prometheus_text()

    @app.get("/corpus", dependencies=[Depends(_guard)])
    def corpus_info():
        return {
            "docs": state.n_docs,
            "chunks": state.n_chunks,
            "loaded_at": state.loaded_at,
            "embedding_model": state.settings.retrieval.embedding_model,
        }

    @app.post("/corpus/load", dependencies=[Depends(_guard)])
    def load_corpus(req: LoadCorpusRequest):
        for i, d in enumerate(req.documents):
            if "text" not in d or "id" not in d:
                raise HTTPException(422, f"documents[{i}] must have 'id' and 'text'")
        _build_pipeline(state, req.documents)
        return {"status": "loaded", "docs": state.n_docs, "chunks": state.n_chunks}

    @app.post("/ask", dependencies=[Depends(_guard)])
    def ask(req: AskRequest):
        # Lazy corpus load: deferred from startup to keep the boot peak
        # under 512 MB on the Render free tier.
        if state.pipeline is None and state._corpus_path is not None:
            import gc
            import json
            log.info("First /ask — loading corpus from %s ...", state._corpus_path.name)
            docs = json.loads(state._corpus_path.read_text(encoding="utf-8"))
            _build_pipeline(state, docs)
            state._corpus_path = None
            gc.collect()
            log.info("Corpus loaded: %d docs, %d chunks", state.n_docs, state.n_chunks)
        if state.pipeline is None:
            raise HTTPException(409, "No corpus loaded. POST /corpus/load first.")
        if not (state.llm and state.llm.api_key):
            raise HTTPException(503, "LLM not configured (missing API key).")
        try:
            result = state.pipeline.ask(req.question, mitigate=req.mitigate)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        metrics.inc("ask_total")
        if result.mitigated:
            metrics.inc("ask_mitigated")
        if result.abstained:
            metrics.inc("ask_abstained")
        return result.to_dict()

    @app.post("/detect", dependencies=[Depends(_guard)])
    def detect(req: DetectRequest):
        if state.nli is None or state.classifier is None:
            raise HTTPException(503, "Detection models not loaded yet.")
        from ..detection.claims import extract_claims

        claims = extract_claims(req.answer, min_chars=state.settings.detection.min_claim_chars)
        scores = state.nli.score_claims(claims, req.passages, batch_size=state.settings.detection.batch_size)
        verdict = state.classifier.verdict(req.answer, scores, len(req.passages), passages=req.passages)
        metrics.inc("detect_total")
        if verdict.hallucinated:
            metrics.inc("detect_flagged")
        return {
            "hallucinated": verdict.hallucinated,
            "probability": verdict.probability,
            "method": verdict.method,
            "n_claims": verdict.n_claims,
            "n_unsupported": verdict.n_unsupported,
            "n_contradicted": verdict.n_contradicted,
            "unsupported_claims": verdict.unsupported_claims,
            "contradicted_claims": verdict.contradicted_claims,
            "claim_details": verdict.claim_details,
        }

    return app


app = create_app()
