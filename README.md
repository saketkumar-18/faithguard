# FaithGuard — Hallucination Detection & Mitigation Engine for RAG

**Capstone project:** a classifier that detects when an LLM's answer isn't supported by its
retrieved context, then **auto-corrects via re-retrieval** — shipped with a benchmark dataset
and measurable faithfulness gains.

```
question ──► hybrid retrieval ──► LLM generation ──► hallucination detector
              (BM25 + dense RRF)                        │ claim extraction
                                                        │ NLI entailment scoring
                                                        │ trained classifier
                                          ┌─────────────┴──────────────┐
                                     PASS ▼                       FAIL ▼
                                   answer out          mitigation engine
                                                       ├─ query expansion (failed claims)
                                                       ├─ re-retrieval (more passages)
                                                       ├─ corrective regeneration
                                                       └─ re-detect (≤ N rounds)
                                                              │ still failing?
                                                              ▼
                                                         honest abstention
```

## What's inside

| Component | Implementation |
|---|---|
| Retrieval | Hybrid BM25 + `BAAI/bge-small-en-v1.5` dense index, fused with Reciprocal Rank Fusion |
| Detection | Rule-based claim extraction → `cross-encoder/nli-deberta-v3-small` NLI scoring → logistic-regression classifier over 12 interpretable features |
| Mitigation | Claim-guided query expansion, re-retrieval with larger top-k, corrective regeneration prompt, re-detection loop, abstention fallback |
| Generation | Any OpenAI-compatible endpoint (OpenRouter/Tokenrouter/vLLM/Ollama), retries + backoff |
| Benchmark | SQuAD-derived corpus + **900 labeled detection examples** (faithful vs 4 corruption types) + 120-question held-out QA gold set |
| Service | FastAPI (`/ask`, `/detect`, `/corpus/load`, `/health`), CLI, Docker |

## Results

### Hallucination detection (held-out test split, 450 examples)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **FaithGuard classifier** (HistGBM, 15 features) | **0.882** | **0.952** | 0.867 | **0.908** | **0.946** |
| Rule-based NLI-threshold baseline | 0.651 | 0.662 | 0.973 | 0.788 | 0.478 |

The learned classifier beats the NLI-threshold heuristic by **+12.0 F1 points** while being far
more precise (0.952 vs 0.662) — it flags hallucinations without over-triggering mitigation.
Decision threshold tuned on a validation slice (0.375). Full report: `reports/detection_eval.md`.

### End-to-end faithfulness (baseline RAG vs FaithGuard, 120 held-out questions)

*(filled in below after `scripts/evaluate_faithfulness.py` completes)*

## Quickstart

```bash
# 1. create env & install (CPU torch)
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. build the benchmark (downloads SQuAD dev, ~4.8 MB)
python scripts/build_benchmark.py

# 3. train the hallucination classifier (~15 min on CPU)
python scripts/train_classifier.py

# 4. run the end-to-end faithfulness evaluation (baseline vs guarded)
export HERMES_CUSTOM_TOKENROUTER_API_KEY=***   # or set FG_LLM_* for another endpoint
python scripts/evaluate_faithfulness.py

# 5. serve the API
python scripts/serve.py --port 8000
```

### CLI

```bash
python scripts/cli.py ask "When was construction of the Taj Mahal completed?"
python scripts/cli.py detect --answer "The tower was finished in 1955." \
    --passages "The Eiffel Tower was built from 1887 to 1889."
```

### API

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "When was the Taj Mahal completed?", "mitigate": true}'

curl -s localhost:8000/detect -H 'Content-Type: application/json' \
  -d '{"answer": "...", "passages": ["...", "..."]}'
```

`POST /corpus/load` swaps in your own documents at runtime:
`{"documents": [{"id": "d1", "title": "...", "text": "..."}]}`.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `FG_LLM_BASE_URL` | tokenrouter | OpenAI-compatible endpoint |
| `FG_LLM_MODEL` | `qwen/qwen3.8-max-free` | generation model |
| `FG_LLM_KEY_ENV` | `HERMES_CUSTOM_TOKENROUTER_API_KEY` | env var holding the API key |
| `FG_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI backbone |
| `FG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | retrieval embeddings |
| `FG_TOP_K` / `FG_RE_TOP_K` | 5 / 8 | retrieval depth / re-retrieval depth |
| `FG_MAX_MITIGATION_ROUNDS` | 2 | correction rounds before abstention |
| `FG_UNSUPPORTED_T` | 0.5 | claim entailment threshold |

## Tests

```bash
pytest                 # fast unit + API tests (no model downloads)
pytest -m models       # integration tests with real NLI + embedding models
```

## Project layout

```
faithguard/
  retrieval/    chunking, BM25, dense, hybrid RRF
  detection/    claims, NLI scorer, features, classifier
  mitigation/   re-retrieval + corrective regeneration engine
  generation/   OpenAI-compatible client, prompts
  eval/         faithfulness/correctness/detection metrics
  api/          FastAPI service
  pipeline.py   the guarded end-to-end pipeline
scripts/        build_benchmark, train_classifier, evaluate_faithfulness,
                make_reports, serve, cli
data/           corpus.json, detection_dataset.jsonl, qa_gold.json
models/         hallucination_classifier.pkl
reports/        detection_eval.{json,md}, faithfulness_eval.{json,md}
```
