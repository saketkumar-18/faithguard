"""Feature engineering: turn claim-level NLI outputs into answer-level features.

The classifier learns the mapping  features -> P(hallucinated answer).
Features are interpretable by design (see FEATURE_NAMES). Two families:

1. NLI features — aggregated entailment/contradiction statistics over claims.
2. Lexical features — token/number overlap between answer, passages, question.
   These catch corruptions NLI is soft on (entity swaps, number perturbations).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .nli import ClaimEvidenceScore

FEATURE_NAMES = [
    # NLI-derived
    "n_claims",                 # number of atomic claims in the answer
    "mean_best_entail",         # mean over claims of best-passage P(entail)
    "min_best_entail",          # weakest claim's best P(entail)
    "p25_best_entail",          # 25th percentile of best P(entail)
    "frac_supported",           # fraction of claims with best P(entail) >= 0.5
    "frac_contradicted",        # fraction of claims contradicted by best passage
    "mean_mean_entail",         # mean over claims of mean-passage P(entail)
    "max_contradiction",        # strongest contradiction seen anywhere
    "frac_hedged",              # fraction of claims containing hedges
    "top1_passage_coverage",    # fraction of claims best-supported by passage 0
    # lexical grounding
    "answer_passage_overlap",   # max token-F1 of answer vs any passage
    "answer_number_overlap",    # fraction of answer numbers present in passages
    "question_answer_overlap",  # token overlap between question and answer
    # shape
    "answer_len_log",           # log(1 + answer char length)
    "claims_per_100_chars",     # claim density
]

_TOKEN = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\b\d{1,4}(?:\.\d+)?\b")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _token_f1(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    common = len(a_tokens & b_tokens)
    if common == 0:
        return 0.0
    p = common / len(a_tokens)
    r = common / len(b_tokens)
    return 2 * p * r / (p + r)


def build_features(
    claim_scores: list[ClaimEvidenceScore],
    answer: str,
    n_passages: int,
    passages: list[str] | None = None,
    question: str | None = None,
) -> np.ndarray:
    """Aggregate claim-level scores + lexical signals into a fixed-size vector."""
    n = len(claim_scores)

    # ---- lexical features (work even with zero claims)
    ans_tok = _tokens(answer)
    if passages:
        overlap = max(_token_f1(ans_tok, _tokens(p)) for p in passages)
        passage_text = " ".join(passages).lower()
        ans_nums = _NUM.findall(answer)
        if ans_nums:
            num_hit = np.mean([1.0 if x in passage_text else 0.0 for x in ans_nums])
        else:
            num_hit = 1.0
    else:
        overlap, num_hit = 0.0, 0.0
    q_overlap = _token_f1(ans_tok, _tokens(question)) if question else 0.0

    if n == 0:
        return np.array(
            [0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             overlap, num_hit, q_overlap,
             np.log1p(len(answer)), 0.0],
            dtype=np.float32,
        )

    best = np.array([c.best_entailment for c in claim_scores], dtype=np.float32)
    contra = np.array([c.contradicted for c in claim_scores], dtype=np.float32)
    mean_ent = np.array([c.mean_entailment for c in claim_scores], dtype=np.float32)
    max_con = max(c.best_contradiction for c in claim_scores)
    hedged = np.array([c.hedged for c in claim_scores], dtype=np.float32)
    top1 = np.array([c.best_passage_idx == 0 for c in claim_scores], dtype=np.float32)

    alen = len(answer)
    return np.array(
        [
            n,
            float(best.mean()),
            float(best.min()),
            float(np.percentile(best, 25)),
            float((best >= 0.5).mean()),
            float(contra.mean()),
            float(mean_ent.mean()),
            float(max_con),
            float(hedged.mean()),
            float(top1.mean()) if n_passages > 0 else 0.0,
            overlap,
            float(num_hit),
            q_overlap,
            float(np.log1p(alen)),
            n / max(alen, 1) * 100.0,
        ],
        dtype=np.float32,
    )
