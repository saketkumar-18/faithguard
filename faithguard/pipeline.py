"""Guarded RAG pipeline: retrieval -> generation -> detection -> mitigation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .detection.classifier import HallucinationClassifier, AnswerVerdict
from .detection.claims import extract_claims
from .detection.nli import NLIScorer
from .generation.llm import LLMClient
from .generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from .mitigation.engine import MitigationEngine, MitigationResult, is_insufficient
from .retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


@dataclass
class GuardedResult:
    question: str
    answer: str
    hallucination_probability: float
    hallucinated_initial: bool
    mitigated: bool
    abstained: bool
    mitigation_rounds: int
    n_claims: int
    n_unsupported: int
    n_contradicted: int
    passages: list[dict]
    verdict: dict
    mitigation_detail: dict | None = None
    latency_ms: float = 0.0
    detection_only: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "hallucination_probability": self.hallucination_probability,
            "hallucinated_initial": self.hallucinated_initial,
            "mitigated": self.mitigated,
            "abstained": self.abstained,
            "mitigation_rounds": self.mitigation_rounds,
            "n_claims": self.n_claims,
            "n_unsupported": self.n_unsupported,
            "n_contradicted": self.n_contradicted,
            "passages": self.passages,
            "verdict": self.verdict,
            "mitigation_detail": self.mitigation_detail,
            "latency_ms": round(self.latency_ms, 1),
            "detection_only": self.detection_only,
        }


class GuardedRAGPipeline:
    """The full FaithGuard pipeline over a fixed corpus.

    Parameters
    ----------
    retriever: built over the corpus chunks.
    llm: LLMClient (may be None for detection-only mode).
    nli: NLIScorer.
    classifier: HallucinationClassifier (may have no trained model -> rules).
    settings: Settings.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient | None,
        nli: NLIScorer,
        classifier: HallucinationClassifier,
        settings: Settings | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.nli = nli
        self.classifier = classifier
        self.settings = settings or get_settings()
        self.mitigator = MitigationEngine(
            retriever=retriever, llm=llm, detector=self.detect, settings=self.settings
        ) if llm is not None else None

    # ------------------------------------------------------------- detection
    def detect(self, answer: str, passages: list[str], question: str | None = None) -> AnswerVerdict:
        """Run claim extraction + NLI + classifier on an answer."""
        dcfg = self.settings.detection
        claims = extract_claims(answer, min_chars=dcfg.min_claim_chars)
        claim_scores = self.nli.score_claims(claims, passages, batch_size=dcfg.batch_size)
        return self.classifier.verdict(
            answer=answer,
            claim_scores=claim_scores,
            n_passages=len(passages),
            passages=passages,
            question=question,
        )

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, mitigate: bool = True) -> GuardedResult:
        t0 = time.perf_counter()
        passages = self.retriever.search(question, top_k=self.settings.retrieval.top_k)
        passage_texts = [p.text for p in passages]
        titles = [p.title for p in passages]

        if self.llm is None:
            raise RuntimeError("No LLM configured; use detect_answer() for detection-only mode")

        prompt = build_rag_prompt(question, passage_texts, titles)
        resp = self.llm.chat(SYSTEM_PROMPT, prompt)
        if not resp.ok:
            raise RuntimeError(f"LLM generation failed: {resp.error}")
        answer = resp.text
        if is_insufficient(answer):
            answer = self.settings.mitigation.abstain_text

        verdict = self.detect(answer, passage_texts, question)
        initial_hallucinated = verdict.hallucinated

        mitigation_detail = None
        mitigated = False
        abstained = False
        rounds = 0
        final_answer = answer
        final_verdict = verdict
        final_passages = passages

        if initial_hallucinated and mitigate and self.mitigator is not None:
            mres: MitigationResult = self.mitigator.mitigate(
                question=question,
                first_answer=answer,
                first_verdict=verdict,
                first_passages=passage_texts,
            )
            final_answer = mres.final_answer
            final_verdict = mres.verdicts[-1]
            mitigation_detail = mres.to_dict()
            mitigated = not mres.abstained and mres.rounds_used > 0
            abstained = mres.abstained
            rounds = mres.rounds_used

        latency = (time.perf_counter() - t0) * 1000.0
        return GuardedResult(
            question=question,
            answer=final_answer,
            hallucination_probability=final_verdict.probability,
            hallucinated_initial=initial_hallucinated,
            mitigated=mitigated,
            abstained=abstained,
            mitigation_rounds=rounds,
            n_claims=final_verdict.n_claims,
            n_unsupported=final_verdict.n_unsupported,
            n_contradicted=final_verdict.n_contradicted,
            passages=[
                {
                    "text": p.text,
                    "doc_id": p.doc_id,
                    "chunk_id": p.chunk_id,
                    "score": round(p.score, 6),
                    "title": p.title,
                }
                for p in final_passages
            ],
            verdict=final_verdict.to_dict(),
            mitigation_detail=mitigation_detail,
            latency_ms=latency,
        )

    # ------------------------------------------------------- detection-only
    def detect_answer(self, answer: str, passages: list[str]) -> GuardedResult:
        """Score a pre-generated answer against given passages (no LLM calls)."""
        t0 = time.perf_counter()
        verdict = self.detect(answer, passages)
        latency = (time.perf_counter() - t0) * 1000.0
        return GuardedResult(
            question="",
            answer=answer,
            hallucination_probability=verdict.probability,
            hallucinated_initial=verdict.hallucinated,
            mitigated=False,
            abstained=False,
            mitigation_rounds=0,
            n_claims=verdict.n_claims,
            n_unsupported=verdict.n_unsupported,
            n_contradicted=verdict.n_contradicted,
            passages=[{"text": p} for p in passages],
            verdict=verdict.to_dict(),
            latency_ms=latency,
            detection_only=True,
        )
