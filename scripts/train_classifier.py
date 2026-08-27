#!/usr/bin/env python
"""Train the hallucination detection classifier on the labeled benchmark.

Reads data/detection_dataset.jsonl, runs NLI scoring over each example's
(answer, passages) pair, builds answer-level features, and trains a
logistic-regression classifier on the TRAIN split. Evaluates on the TEST
split and saves:
  - models/hallucination_classifier.pkl
  - reports/detection_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithguard.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, get_settings
from faithguard.detection.claims import extract_claims
from faithguard.detection.features import build_features, FEATURE_NAMES
from faithguard.detection.nli import NLIScorer
from faithguard.detection.classifier import HallucinationClassifier
from faithguard.eval.metrics import detection_report


def featurize_examples(examples: list[dict], nli: NLIScorer, settings, desc: str) -> tuple[np.ndarray, np.ndarray]:
    from tqdm import tqdm

    X, y = [], []
    for ex in tqdm(examples, desc=desc):
        claims = extract_claims(ex["answer"], min_chars=settings.detection.min_claim_chars)
        scores = nli.score_claims(claims, ex["passages"], batch_size=settings.detection.batch_size)
        feats = build_features(
            scores, ex["answer"], len(ex["passages"]),
            passages=ex["passages"], question=ex.get("question"),
        )
        X.append(feats)
        y.append(ex["label"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATA_DIR / "detection_dataset.jsonl"))
    ap.add_argument("--out-model", default=str(MODELS_DIR / "hallucination_classifier.pkl"))
    ap.add_argument("--out-report", default=str(REPORTS_DIR / "detection_eval.json"))
    ap.add_argument("--max-train", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    settings = get_settings()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    examples = [json.loads(l) for l in open(args.dataset, encoding="utf-8")]
    train = [e for e in examples if e["split"] == "train"]
    test = [e for e in examples if e["split"] == "test"]
    if args.max_train:
        train = train[: args.max_train]
    print(f"[train] {len(train)} train / {len(test)} test examples")

    print(f"[train] loading NLI model {settings.detection.nli_model} ...")
    t0 = time.time()
    nli = NLIScorer(settings.detection.nli_model, device=settings.device)
    print(f"[train] NLI model loaded in {time.time()-t0:.1f}s")

    X_train, y_train = featurize_examples(train, nli, settings, "featurize train")
    X_test, y_test = featurize_examples(test, nli, settings, "featurize test")

    clf = HallucinationClassifier()
    t0 = time.time()
    train_meta = clf.fit(X_train, y_train)
    print(f"[train] classifier trained in {time.time()-t0:.1f}s  meta={train_meta}")

    # evaluate on test
    proba = np.array([clf.predict_proba(x) for x in X_test])
    report = detection_report(y_test.tolist(), proba.tolist())

    # also evaluate the rule-based fallback for comparison
    rule_clf = HallucinationClassifier()  # no model -> rules
    rule_probs = []
    for ex in test:
        claims = extract_claims(ex["answer"], min_chars=settings.detection.min_claim_chars)
        scores = nli.score_claims(claims, ex["passages"], batch_size=settings.detection.batch_size)
        v = rule_clf.verdict(ex["answer"], scores, len(ex["passages"]),
                             passages=ex["passages"], question=ex.get("question"))
        rule_probs.append(v.probability)
    rule_report = detection_report(y_test.tolist(), rule_probs)

    # feature importance (permutation importance of the gradient boosting model)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(clf.model, X_test, y_test, n_repeats=5, random_state=42)
    importance = sorted(zip(FEATURE_NAMES, [float(c) for c in perm.importances_mean]),
                        key=lambda kv: abs(kv[1]), reverse=True)

    out = {
        "model": settings.detection.nli_model,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_metrics": train_meta,
        "test_classifier": report,
        "test_rule_fallback": rule_report,
        "feature_importance": [{"feature": f, "coef": c} for f, c in importance],
        "train_time_s": round(time.time() - t0, 2),
    }
    Path(args.out_report).write_text(json.dumps(out, indent=2), encoding="utf-8")
    clf.save(args.out_model, meta=out)
    print(f"[train] test F1={report['f1']:.3f}  AUC={report['auc']:.3f}  "
          f"(rules baseline F1={rule_report['f1']:.3f})")
    print(f"[train] saved {args.out_model}")
    print(f"[train] saved {args.out_report}")


if __name__ == "__main__":
    main()
