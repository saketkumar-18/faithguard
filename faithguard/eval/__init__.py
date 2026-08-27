"""Evaluation: faithfulness metrics for RAG answers.

Metrics
-------
- claim_precision (faithfulness): fraction of answer claims entailed by context.
- answer_faithfulness: 1.0 if all claims supported, else fraction supported.
- answer_correctness: token-F1 of the answer vs the gold reference answer.
- detection metrics: precision/recall/F1/AUC of the hallucination classifier
  against labeled (answer, context, label) examples.
"""
from .metrics import (
    token_f1,
    answer_correctness,
    claim_precision,
    answer_faithfulness,
    detection_report,
)

__all__ = [
    "token_f1",
    "answer_correctness",
    "claim_precision",
    "answer_faithfulness",
    "detection_report",
]
