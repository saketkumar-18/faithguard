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
| Detection | Rule-based claim extraction → `cross-encoder/nli-deberta-v3-small` NLI scoring → gradient-boosting classifier (HistGBM) over 15 interpretable features, threshold tuned on validation |
| Mitigation | Claim-guided query expansion, re-retrieval with larger top-k, corrective regeneration prompt, re-detection loop, abstention fallback |
| Generation | Any OpenAI-compatible endpoint (OpenRouter/Tokenrouter/vLLM/Ollama), retries + backoff |
| Benchmark | SQuAD-derived corpus + **900 labeled detection examples** (faithful vs 4 corruption types) + 120-question held-out QA gold set |
| Service | FastAPI (`/ask`, `/detect`, `/corpus/load`, `/health`), CLI, Docker |

## Results

### Hallucination detection (held-out test split, 450 examples)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **FaithGuard classifier** (HistGBM, 15 features) | **0.856** | **0.930** | 0.847 | **0.887** | **0.948** |
| Rule-based NLI-threshold baseline | 0.616 | 0.662 | 0.867 | 0.750 | 0.549 |

The learned classifier beats the NLI-threshold heuristic by **+13.7 F1 points** while being far
more precise (0.930 vs 0.662) — it flags hallucinations without over-triggering mitigation.
Decision threshold tuned on a validation slice (0.50). Full report: `reports/detection_eval.md`.

### End-to-end faithfulness (baseline RAG vs FaithGuard, 120 held-out questions)

| Metric | Baseline RAG | FaithGuard | Delta |
|---|---|---|---|
| **Faithfulness** (mean soft support) | 0.497 | **0.620** | **+0.123** |
| **Claim precision** (fraction of claims supported) | 0.161 | **0.375** | **+0.214** |
| Answer containment (gold answer present) | 0.867 | 0.783 | −0.083 |
| Answer correctness (token-F1) | 0.743 | 0.561 | −0.182 |
| Abstention rate | 0.000 | 0.133 | — |
| Mean latency (ms) | 6,040 | 21,666 | — |

**Where the gains come from.** On the 46 questions the detector flagged as hallucinated,
mitigation (claim-guided re-retrieval + corrective regeneration) raised mean faithfulness
from **0.416 → 0.705**; 30 of 46 were fully repaired and 34/46 ended with faithfulness ≥ 0.5.

**The honest trade-off.** FaithGuard abstains on 13.3% of questions (16/120) when it cannot
verify an answer after re-retrieval — abstention is faithful by construction but scores 0 on
correctness. Excluding abstentions, the guarded pipeline actually *improves* answer
containment (0.885 → 0.904) and faithfulness (0.504 → 0.561). The token-F1 drop is largely a
metric artifact: the corrective prompt elicits full-sentence answers ("X happened in 1992")
which token-F1 penalizes against fragment gold answers ("1992") even when the answer is
present and correct — containment is the fairer measure here.

Full report: `reports/faithfulness_eval.md`.

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

## Production deployment

FaithGuard ships production-hardened. The same image runs in local dev and in
the cloud — the difference is a few environment variables.

### Security & operations (set these before going public)

| Variable | Purpose |
|---|---|
| `FG_API_KEY` | **Enable API-key auth.** Send it via the `x-api-key` header or as a Bearer token in the `Authorization` header. When unset, auth is OFF and the app logs a loud warning. `/health` and `/metrics` stay open for load balancers. |
| `FG_RATE_LIMIT` | Max requests per window per key (e.g. `60`). Unset = no limit. |
| `FG_RATE_WINDOW_S` | Rate-limit window in seconds (default `60`). |
| `FG_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` (default `INFO`). |

Every request gets an `x-request-id` (echoed back in the response) and a
structured access-log line. `GET /metrics` exposes Prometheus-style counters
(requests, latency, mitigation/abstention/detection tallies).

### Run with Docker

```bash
# simplest: compose (reads .env, persists models, restarts automatically)
cp .env.example .env   # fill in FG_API_KEY + LLM key
docker compose up -d
docker compose logs -f

# or manually:
docker build -t faithguard .
docker run -p 8000:8000 \
  -e FG_API_KEY=*** \
  -e FG_RATE_LIMIT=60 \
  -e HERMES_CUSTOM_TOKENROUTER_API_KEY=*** \
  -v faithguard-models:/models-cache \
  faithguard
```

The image runs as a **non-root user**, has a health check, and shuts down
gracefully (30s drain). Models download on first boot and persist in the
`/models-cache` volume.

### Deploy to Render

`render.yaml` is included — connect the repo in Render and it builds the
Dockerfile automatically. Add the env vars above in the dashboard (mark the
API keys as secret).

### Operational notes

- **Single worker.** The NLI + embedding models live in-process; keep
  `--workers 1` and scale horizontally (multiple instances) if you need more
  throughput. The rate limiter is per-instance (in-memory); use Redis if you
  scale out and need a global budget.
- **LLM degradation.** If the generation endpoint is down, `/ask` returns 502
  but `/detect` (detection-only, no LLM) keeps working — you can still score
  answers from another generator.
- **Cold start.** First boot downloads ~200 MB of models. The baked-in
  embedding cache means the corpus is not re-embedded.

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
