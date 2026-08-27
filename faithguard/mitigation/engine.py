"""Mitigation engine: when an answer is flagged, re-retrieve and regenerate.

Strategy per round:
1. Build an expanded query: original question + the unsupported/contradicted
   claims (they name exactly the entities/facts the first retrieval missed).
2. Retrieve more passages (re_top_k > top_k) with the expanded query.
3. Regenerate with a corrective prompt that explicitly lists the bad claims.
4. Re-run detection on the new answer. Stop when it passes or rounds run out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..detection.classifier import AnswerVerdict

log = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    final_answer: str
    rounds_used: int
    abstained: bool
    verdicts: list[AnswerVerdict] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    n_passages_final: int = 0

    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "rounds_used": self.rounds_used,
            "abstained": self.abstained,
            "n_passages_final": self.n_passages_final,
            "queries_used": self.queries_used,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


class MitigationEngine:
    def __init__(self, retriever, llm, detector, settings):
        """
        retriever: HybridRetriever with .search(query, top_k)
        llm: LLMClient with .chat(system, user)
        detector: callable(answer, passages, question) -> AnswerVerdict
        settings: faithguard.config.Settings
        """
        self.retriever = retriever
        self.llm = llm
        self.detector = detector
        self.settings = settings

    # ------------------------------------------------------------- query build
    @staticmethod
    def expand_query(question: str, verdict: AnswerVerdict) -> str:
        """Original question + the claims that failed, as retrieval signal."""
        extras = verdict.contradicted_claims + verdict.unsupported_claims
        # keep the expansion focused: at most 3 claims, truncated
        extras = [c[:160] for c in extras[:3]]
        if not extras:
            return question
        return question + " " + " ".join(extras)

    # ------------------------------------------------------------------ run
    def mitigate(
        self,
        question: str,
        first_answer: str,
        first_verdict: AnswerVerdict,
        first_passages: list[str],
    ) -> MitigationResult:
        from ..generation.prompts import SYSTEM_PROMPT, build_corrective_prompt

        mcfg = self.settings.mitigation
        verdicts = [first_verdict]
        queries = [question]
        answer = first_answer
        passages = first_passages
        verdict = first_verdict

        for round_no in range(1, mcfg.max_rounds + 1):
            if not verdict.hallucinated:
                break
            query = self.expand_query(question, verdict)
            queries.append(query)
            new_passages = self.retriever.search(query, top_k=self.settings.retrieval.re_top_k)
            passage_texts = [p.text for p in new_passages]
            titles = [p.title for p in new_passages]

            prompt = build_corrective_prompt(
                question=question,
                passages=passage_texts,
                previous_answer=answer,
                unsupported_claims=verdict.unsupported_claims,
                contradicted_claims=verdict.contradicted_claims,
                titles=titles,
            )
            resp = self.llm.chat(SYSTEM_PROMPT, prompt)
            if not resp.ok or not resp.text:
                log.warning("Mitigation round %d: LLM failed (%s); abstaining", round_no, resp.error)
                return MitigationResult(
                    final_answer=mcfg.abstain_text,
                    rounds_used=round_no,
                    abstained=True,
                    verdicts=verdicts,
                    queries_used=queries,
                    n_passages_final=len(passage_texts),
                )
            answer = _strip_insufficient(resp.text)
            if not answer:
                # model says the re-retrieved context still lacks the answer
                return MitigationResult(
                    final_answer=mcfg.abstain_text,
                    rounds_used=round_no,
                    abstained=True,
                    verdicts=verdicts,
                    queries_used=queries,
                    n_passages_final=len(passage_texts),
                )
            passages = passage_texts
            verdict = self.detector(answer, passages, question)
            verdicts.append(verdict)
            log.info(
                "Mitigation round %d: p(hallucinated)=%.3f hallucinated=%s",
                round_no, verdict.probability, verdict.hallucinated,
            )

        if verdict.hallucinated:
            # still failing after all rounds -> honest abstention
            return MitigationResult(
                final_answer=mcfg.abstain_text,
                rounds_used=mcfg.max_rounds,
                abstained=True,
                verdicts=verdicts,
                queries_used=queries,
                n_passages_final=len(passages),
            )

        return MitigationResult(
            final_answer=answer,
            rounds_used=len(verdicts) - 1,
            abstained=False,
            verdicts=verdicts,
            queries_used=queries,
            n_passages_final=len(passages),
        )


def _strip_insufficient(text: str) -> str:
    t = text.strip()
    if t.upper().startswith("INSUFFICIENT_CONTEXT"):
        return ""
    return t


def is_insufficient(text: str) -> bool:
    return not text or text.strip().upper().startswith("INSUFFICIENT_CONTEXT")
