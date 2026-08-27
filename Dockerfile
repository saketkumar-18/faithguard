# FaithGuard — Hallucination Detection & Mitigation Engine
# CPU-only, torch-free production image. Both ML models are BAKED into the
# image at build time so there is no runtime download (the ~300 MB download
# buffer was what OOM'd the 512 MB Render free tier).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_models \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    FASTEMBED_CACHE_PATH=/app/hf_models/fastembed \
    FAITHGUARD_DATA_DIR=/app/data \
    FAITHGUARD_MODELS_DIR=/app/models \
    # production logging
    FG_LOG_LEVEL=INFO

WORKDIR /app

# Torch-free runtime: ONNX Runtime + fastembed keep the image small and the
# container under 512 MB RAM (Render free tier).
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- bake the ML models into the image (no runtime download) ---------------
# NLI cross-encoder (quantized int8 ONNX). The embedding model is NOT baked:
# on the 512 MB free tier we run BM25-only retrieval (FG_USE_DENSE=0), so the
# ~240 MB fastembed model is never loaded.
# HF_HUB_OFFLINE=0 overrides the global offline flag for this build step only.
RUN HF_HUB_OFFLINE=0 python -c "from huggingface_hub import hf_hub_download; \
[hf_hub_download('Xenova/nli-deberta-v3-small', f) for f in \
 ['onnx/model_quantized.onnx', 'tokenizer.json', 'config.json']]"

COPY faithguard ./faithguard
COPY scripts ./scripts
COPY pyproject.toml .

# Bake the benchmark corpus + embedding cache + trained classifier into the
# image (fast cold starts, no re-embedding). Globs tolerate absent files.
COPY data/corpus.json* ./data/
COPY data/embed_cache.npz* ./data/
COPY models/*.pkl* ./models/

# --- security: run as a non-root user -------------------------------------
RUN groupadd --system faithguard && useradd --system --gid faithguard \
        --home-dir /app --shell /usr/sbin/nologin faithguard \
    && chown -R faithguard:faithguard /app
USER faithguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=5)" || exit 1

# --timeout-graceful-shutdown lets in-flight requests finish on SIGTERM.
# Render injects $PORT (default 8000 locally).
CMD uvicorn faithguard.api.app:app \
     --host 0.0.0.0 --port "${PORT:-8000}" \
     --workers 1 \
     --timeout-graceful-shutdown 30 \
     --access-log
