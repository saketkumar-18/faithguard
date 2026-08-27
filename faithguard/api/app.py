"""FastAPI app: guarded RAG over a loaded corpus, plus detection-only scoring.

Endpoints
---------
GET  /health                 liveness + model status
GET  /corpus                 corpus stats
POST /ask                    full guarded pipeline (retrieve+generate+detect+mitigate)
POST /detect                 score a pre-generated answer against given passages
POST /corpus/load            load a new corpus at runtime (list of documents)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..config import DATA_DIR, MODELS_DIR, get_settings
from ..detection.classifier import HallucinationClassifier
from ..detection.nli import NLIScorer
from ..generation.llm import LLMClient
from ..pipeline import GuardedRAGPipeline
from ..retrieval.chunking import Chunker
from ..retrieval.hybrid import HybridRetriever

log = logging.getLogger("faithguard.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mitigate: bool = True
    top_k: int | None = Field(default=None, ge=1, le=20)


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
    )
    state.pipeline = GuardedRAGPipeline(retriever, state.llm, state.nli, state.classifier, s)
    state.n_chunks = len(chunks)
    state.n_docs = len(documents)
    state.loaded_at = time.time()


def create_app(load_default_corpus: bool = True) -> FastAPI:
    state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.time()
        log.info("Loading NLI model %s ...", state.settings.detection.nli_model)
        state.nli = NLIScorer(state.settings.detection.nli_model, device=state.settings.device)
        state.classifier = HallucinationClassifier(
            MODELS_DIR / "hallucination_classifier.pkl",
            unsupported_threshold=state.settings.detection.unsupported_claim_threshold,
            hallucinated_fraction=state.settings.detection.hallucinated_answer_fraction,
        )
        state.llm = LLMClient.from_settings(state.settings)
        if load_default_corpus:
            corpus_path = DATA_DIR / "corpus.json"
            if corpus_path.exists():
                import json
                docs = json.loads(corpus_path.read_text(encoding="utf-8"))
                _build_pipeline(state, docs)
                log.info("Corpus loaded: %d docs, %d chunks", state.n_docs, state.n_chunks)
            else:
                log.warning("No corpus.json found; call POST /corpus/load before /ask")
        log.info("FaithGuard API ready in %.1fs", time.time() - t0)
        yield

    app = FastAPI(
        title="FaithGuard",
        description="Hallucination Detection & Mitigation Engine for RAG",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.fg = state

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

    @app.get("/corpus")
    def corpus_info():
        return {
            "docs": state.n_docs,
            "chunks": state.n_chunks,
            "loaded_at": state.loaded_at,
            "embedding_model": state.settings.retrieval.embedding_model,
        }

    @app.post("/corpus/load")
    def load_corpus(req: LoadCorpusRequest):
        for i, d in enumerate(req.documents):
            if "text" not in d or "id" not in d:
                raise HTTPException(422, f"documents[{i}] must have 'id' and 'text'")
        _build_pipeline(state, req.documents)
        return {"status": "loaded", "docs": state.n_docs, "chunks": state.n_chunks}

    @app.post("/ask")
    def ask(req: AskRequest):
        if state.pipeline is None:
            raise HTTPException(409, "No corpus loaded. POST /corpus/load first.")
        if not (state.llm and state.llm.api_key):
            raise HTTPException(503, "LLM not configured (missing API key).")
        if req.top_k:
            # temporary override via settings is not frozen-safe; search directly instead
            pass
        try:
            result = state.pipeline.ask(req.question, mitigate=req.mitigate)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        return result.to_dict()

    @app.post("/detect")
    def detect(req: DetectRequest):
        if state.nli is None or state.classifier is None:
            raise HTTPException(503, "Detection models not loaded yet.")
        from ..detection.claims import extract_claims

        claims = extract_claims(req.answer, min_chars=state.settings.detection.min_claim_chars)
        scores = state.nli.score_claims(claims, req.passages, batch_size=state.settings.detection.batch_size)
        verdict = state.classifier.verdict(req.answer, scores, len(req.passages), passages=req.passages)
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
