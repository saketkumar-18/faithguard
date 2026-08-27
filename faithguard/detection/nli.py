"""NLI-based claim verification against retrieved evidence.

Uses a cross-encoder NLI model (default: cross-encoder/nli-deberta-v3-small)
to score each (claim, evidence_passage) pair into
P(entailment), P(neutral), P(contradiction).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClaimEvidenceScore:
    claim: str
    hedged: bool
    best_entailment: float          # max P(entail) over passages
    best_contradiction: float       # P(contradiction) at the best-entailing passage
    mean_entailment: float          # mean P(entail) over passages
    best_passage_idx: int           # index of the passage that best supports it
    per_passage_entailment: list[float] = field(default_factory=list)
    best_neutral: float = 0.0       # P(neutral) at the best-entailing passage

    @property
    def supported(self) -> bool:
        return self.best_entailment >= 0.5

    @property
    def contradicted(self) -> bool:
        return self.best_contradiction > self.best_entailment and self.best_contradiction >= 0.4

    @property
    def support(self) -> float:
        """Soft support in [0,1]: full credit for entailment, half credit for
        neutral (consistent but not strictly entailed — paraphrase / minor
        inference). Contradictions get none. NLI cross-encoders are strict:
        a paraphrased-but-correct claim usually lands in `neutral`, so raw
        entailment alone saturates at the floor for real RAG answers."""
        return float(self.best_entailment + 0.5 * self.best_neutral)


class NLIScorer:
    """Batched NLI scoring of claims against evidence passages."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        # nli-deberta-v3-* label order: [contradiction, entailment, neutral]
        labels = list(getattr(self.model, "config", None) and self.model.config.id2label.values() or [])
        labels = [l.lower() for l in labels]
        if "entailment" in labels:
            self._ent = labels.index("entailment")
            self._con = labels.index("contradiction") if "contradiction" in labels else 0
            self._neu = labels.index("neutral") if "neutral" in labels else 2
        else:  # fallback to common ordering
            self._con, self._ent, self._neu = 0, 1, 2

    def score_claims(
        self,
        claims: list[dict],
        passages: list[str],
        batch_size: int = 32,
    ) -> list[ClaimEvidenceScore]:
        """Score every claim against every passage (batched)."""
        if not claims or not passages:
            return [
                ClaimEvidenceScore(c["text"], c.get("hedged", False), 0.0, 0.0, 0.0, -1, [0.0] * len(passages))
                for c in claims
            ]
        # NLI cross-encoders expect (premise, hypothesis) = (passage, claim).
        # Reversed order collapses entailment to ~0 (model sees the claim as
        # the premise) — keep this order!
        pairs = [(p, c["text"]) for c in claims for p in passages]
        raw = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:  # single pair
            raw = raw.reshape(1, -1)
        probs = _softmax(raw)

        out: list[ClaimEvidenceScore] = []
        n_p = len(passages)
        for i, c in enumerate(claims):
            block = probs[i * n_p : (i + 1) * n_p]
            ent = block[:, self._ent]
            con = block[:, self._con]
            neu = block[:, self._neu]
            best = int(np.argmax(ent))
            out.append(
                ClaimEvidenceScore(
                    claim=c["text"],
                    hedged=c.get("hedged", False),
                    best_entailment=float(ent[best]),
                    best_contradiction=float(con[best]),
                    mean_entailment=float(ent.mean()),
                    best_passage_idx=best,
                    per_passage_entailment=[float(x) for x in ent],
                    best_neutral=float(neu[best]),
                )
            )
        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
