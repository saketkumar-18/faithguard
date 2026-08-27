"""Faithfulness and correctness metrics."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between prediction and reference."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred)
    recall = n_same / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(prediction: str, references: list[str]) -> float:
    """Max token-F1 over a list of acceptable reference answers."""
    if not references:
        return 0.0
    return max(token_f1(prediction, r) for r in references)


def claim_precision(claim_scores: list) -> float:
    """Faithfulness at claim level: fraction of claims entailed by context."""
    if not claim_scores:
        return 1.0
    return float(np.mean([c.supported for c in claim_scores]))


def answer_faithfulness(claim_scores: list) -> float:
    """Answer-level faithfulness in [0,1]: mean soft support across claims.

    Soft support = P(entailment) + 0.5 * P(neutral): full credit for strict
    entailment, half credit for claims that are consistent with the context
    but need paraphrase/minor inference (NLI cross-encoders are strict, so
    raw entailment alone saturates near zero for real RAG answers).
    Contradicted claims contribute nothing.
    """
    if not claim_scores:
        return 1.0
    vals = []
    for c in claim_scores:
        if getattr(c, "contradicted", False):
            vals.append(0.0)
        else:
            vals.append(getattr(c, "support", c.best_entailment))
    return float(np.mean(vals))


def detection_report(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    """Precision/recall/F1/AUC/accuracy for hallucination detection."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    out = {
        "n": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return out
