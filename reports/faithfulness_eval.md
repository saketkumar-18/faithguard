# Faithfulness Evaluation — Baseline RAG vs FaithGuard

- Questions: 120 (held-out SQuAD articles)
- LLM: `qwen/qwen3.8-max-free` | NLI: `cross-encoder/nli-deberta-v3-small` | Embeddings: `BAAI/bge-small-en-v1.5`
- Retrieval top_k=5, re-retrieval top_k=8, max mitigation rounds=2

## Headline results

| Metric | Baseline RAG | FaithGuard | Delta |
|---|---|---|---|
| Faithfulness (mean soft support) | 0.497 | 0.620 | +0.123 |
| Claim precision (fraction supported) | 0.161 | 0.375 | +0.214 |
| Answer containment (gold answer present) | 0.867 | 0.783 | -0.083 |
| Answer correctness (token-F1 vs gold) | 0.743 | 0.561 | -0.182 |
| Hallucination rate (flagged) | 0.325 | 0.383 | +0.058 |
| Abstention rate | 0.000 | 0.133 | — |
| Mean latency (ms) | 6040 | 21666 | — |

Mitigation fired on 30 questions.
