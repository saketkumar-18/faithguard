"""Trained hallucination classifier over NLI-derived + lexical features.

Two layers:
1. A gradient-boosting classifier (scikit-learn HistGradientBoosting) trained
   on the labeled benchmark dataset, with a decision threshold tuned on a
   validation split for best F1. Learns features -> P(hallucinated).
2. A rule-based fallback (threshold logic) used when no trained model is
   available, so the engine works out of the box.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES
from .nli import ClaimEvidenceScore


@dataclass
class AnswerVerdict:
    hallucinated: bool
    probability: float
    method: str                     # "classifier" | "rules"
    n_claims: int
    n_unsupported: int
    n_contradicted: int
    unsupported_claims: list[str]
    contradicted_claims: list[str]
    claim_details: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class HallucinationClassifier:
    """Wraps the trained sklearn model + rule fallback and produces verdicts."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        unsupported_threshold: float = 0.5,
        hallucinated_fraction: float = 0.35,
    ):
        self.unsupported_threshold = unsupported_threshold
        self.hallucinated_fraction = hallucinated_fraction
        self.model = None
        self.decision_threshold = 0.5
        self.model_meta: dict = {}
        if model_path and Path(model_path).exists():
            self.load(model_path)

    # ------------------------------------------------------------------ model
    def load(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.model_meta = payload.get("meta", {})
        self.decision_threshold = float(self.model_meta.get("decision_threshold", 0.5))

    def save(self, model_path: str | Path, meta: dict | None = None) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "meta": meta or {}}, f)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
        """Train HistGradientBoosting + tune the decision threshold.

        Threshold is tuned on a stratified 25% validation slice for best F1,
        then the model is refit on all data with that threshold.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed
        )

        def make_model():
            return HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_depth=4,
                l2_regularization=1.0,
                min_samples_leaf=10,
                random_state=seed,
            )

        m = make_model()
        m.fit(X_tr, y_tr)
        val_proba = m.predict_proba(X_val)[:, 1]

        # tune threshold on validation slice
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.15, 0.86, 0.025):
            f1 = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        # refit on all data
        final = make_model()
        final.fit(X, y)
        self.model = final
        self.decision_threshold = best_t

        full_proba = final.predict_proba(X)[:, 1]
        meta = {
            "model_type": "HistGradientBoostingClassifier",
            "feature_names": FEATURE_NAMES,
            "n_train": int(len(y)),
            "decision_threshold": best_t,
            "val_f1_at_tuned_threshold": float(best_f1),
            "train_f1": float(f1_score(y, (full_proba >= best_t).astype(int), zero_division=0)),
        }
        self.model_meta = meta
        return meta

    # ---------------------------------------------------------------- predict
    def predict_proba(self, features: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("No trained model loaded")
        return float(self.model.predict_proba(features.reshape(1, -1))[0, 1])

    def verdict(
        self,
        answer: str,
        claim_scores: list[ClaimEvidenceScore],
        n_passages: int,
        passages: list[str] | None = None,
        question: str | None = None,
        decision_threshold: float | None = None,
    ) -> AnswerVerdict:
        from .features import build_features

        unsupported = [c for c in claim_scores if c.best_entailment < self.unsupported_threshold]
        contradicted = [c for c in claim_scores if c.contradicted]

        if self.model is not None:
            feats = build_features(claim_scores, answer, n_passages, passages, question)
            prob = self.predict_proba(feats)
            method = "classifier"
        else:
            prob = self._rule_probability(claim_scores)
            method = "rules"

        threshold = decision_threshold if decision_threshold is not None else self.decision_threshold
        return AnswerVerdict(
            hallucinated=bool(prob >= threshold),
            probability=round(prob, 4),
            method=method,
            n_claims=len(claim_scores),
            n_unsupported=len(unsupported),
            n_contradicted=len(contradicted),
            unsupported_claims=[c.claim for c in unsupported],
            contradicted_claims=[c.claim for c in contradicted],
            claim_details=[
                {
                    "claim": c.claim,
                    "best_entailment": round(c.best_entailment, 4),
                    "best_contradiction": round(c.best_contradiction, 4),
                    "best_neutral": round(getattr(c, "best_neutral", 0.0), 4),
                    "mean_entailment": round(c.mean_entailment, 4),
                    "best_passage_idx": c.best_passage_idx,
                    "hedged": c.hedged,
                    "supported": c.supported,
                    "contradicted": c.contradicted,
                }
                for c in claim_scores
            ],
        )

    def _rule_probability(self, claim_scores: list[ClaimEvidenceScore]) -> float:
        """Calibrated-ish rule fallback when no trained model exists."""
        if not claim_scores:
            return 0.1
        best = np.array([c.best_entailment for c in claim_scores])
        frac_unsup = float((best < self.unsupported_threshold).mean())
        frac_contra = float(np.array([c.contradicted for c in claim_scores]).mean())
        # weighted blend: unsupported fraction dominates, contradictions add risk
        p = 0.75 * frac_unsup + 0.45 * frac_contra
        return float(min(1.0, max(0.0, p)))
