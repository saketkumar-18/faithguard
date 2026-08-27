# Hallucination Detection — Classifier Evaluation

- NLI backbone: `cross-encoder/nli-deberta-v3-small`
- Train examples: 450 | Test examples: 450

## Test-set results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Trained classifier | 0.856 | 0.930 | 0.847 | 0.887 | 0.948 |
| Rule-based fallback | 0.616 | 0.662 | 0.867 | 0.750 | 0.549 |

Confusion (trained): TP=254 FP=19 FN=46 TN=131

## Feature importance (permutation importance on test set)

| Feature | Importance |
|---|---|
| answer_passage_overlap | +0.160 |
| min_best_entail | +0.115 |
| answer_len_log | +0.093 |
| mean_best_entail | -0.012 |
| answer_number_overlap | +0.007 |
| max_contradiction | +0.005 |
| claims_per_100_chars | +0.004 |
| p25_best_entail | -0.003 |
| mean_mean_entail | +0.003 |
| n_claims | -0.001 |
| top1_passage_coverage | -0.001 |
| frac_contradicted | -0.000 |
| question_answer_overlap | +0.000 |
| frac_supported | +0.000 |
| frac_hedged | +0.000 |
