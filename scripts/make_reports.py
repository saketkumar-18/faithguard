#!/usr/bin/env python
"""Render markdown reports from the JSON evaluation outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faithguard.config import REPORTS_DIR


def detection_md() -> str:
    d = json.loads((REPORTS_DIR / "detection_eval.json").read_text(encoding="utf-8"))
    tc, rf = d["test_classifier"], d["test_rule_fallback"]
    lines = [
        "# Hallucination Detection — Classifier Evaluation",
        "",
        f"- NLI backbone: `{d['model']}`",
        f"- Train examples: {d['n_train']} | Test examples: {d['n_test']}",
        "",
        "## Test-set results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | AUC |",
        "|---|---|---|---|---|---|",
        f"| Trained classifier | {tc['accuracy']:.3f} | {tc['precision']:.3f} | {tc['recall']:.3f} | {tc['f1']:.3f} | {tc['auc']:.3f} |",
        f"| Rule-based fallback | {rf['accuracy']:.3f} | {rf['precision']:.3f} | {rf['recall']:.3f} | {rf['f1']:.3f} | {rf['auc']:.3f} |",
        "",
        f"Confusion (trained): TP={tc['confusion']['tp']} FP={tc['confusion']['fp']} "
        f"FN={tc['confusion']['fn']} TN={tc['confusion']['tn']}",
        "",
        "## Feature importance (permutation importance on test set)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for fi in d["feature_importance"]:
        lines.append(f"| {fi['feature']} | {fi['coef']:+.3f} |")
    return "\n".join(lines) + "\n"


def faithfulness_md() -> str:
    d = json.loads((REPORTS_DIR / "faithfulness_eval.json").read_text(encoding="utf-8"))
    b, g, gains = d["baseline"], d["guarded"], d["gains"]
    cfg = d["config"]
    lines = [
        "# Faithfulness Evaluation — Baseline RAG vs FaithGuard",
        "",
        f"- Questions: {d['n_questions']} (held-out SQuAD articles)",
        f"- LLM: `{cfg['llm_model']}` | NLI: `{cfg['nli_model']}` | Embeddings: `{cfg['embedding_model']}`",
        f"- Retrieval top_k={cfg['top_k']}, re-retrieval top_k={cfg['re_top_k']}, max mitigation rounds={cfg['max_mitigation_rounds']}",
        "",
        "## Headline results",
        "",
        "| Metric | Baseline RAG | FaithGuard | Delta |",
        "|---|---|---|---|",
        f"| Faithfulness (mean soft support) | {b['mean_faithfulness']:.3f} | {g['mean_faithfulness']:.3f} | {gains['faithfulness_delta']:+.3f} |",
        f"| Claim precision (fraction supported) | {b['mean_claim_precision']:.3f} | {g['mean_claim_precision']:.3f} | {gains['claim_precision_delta']:+.3f} |",
        f"| Answer containment (gold answer present) | {b.get('mean_containment', 0):.3f} | {g.get('mean_containment', 0):.3f} | {gains.get('containment_delta', 0):+.3f} |",
        f"| Answer correctness (token-F1 vs gold) | {b['mean_correctness']:.3f} | {g['mean_correctness']:.3f} | {gains['correctness_delta']:+.3f} |",
        f"| Hallucination rate (flagged) | {b['hallucination_rate']:.3f} | {g['hallucination_rate']:.3f} | {gains['hallucination_rate_delta']:+.3f} |",
        f"| Abstention rate | {b['abstention_rate']:.3f} | {g['abstention_rate']:.3f} | — |",
        f"| Mean latency (ms) | {b['mean_latency_ms']:.0f} | {g['mean_latency_ms']:.0f} | — |",
        "",
        f"Mitigation fired on {d['n_mitigated']} questions.",
    ]
    return "\n".join(lines) + "\n"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if (REPORTS_DIR / "detection_eval.json").exists():
        out = REPORTS_DIR / "detection_eval.md"
        out.write_text(detection_md(), encoding="utf-8")
        print(f"wrote {out}")
    if (REPORTS_DIR / "faithfulness_eval.json").exists():
        out = REPORTS_DIR / "faithfulness_eval.md"
        out.write_text(faithfulness_md(), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
