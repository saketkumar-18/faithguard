# FaithGuard — Hallucination Detection & Mitigation Engine
# CPU-only production image. Models are downloaded at first startup and
# cached in /models-cache (mount a volume there to persist across restarts).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models-cache \
    FAITHGUARD_DATA_DIR=/app/data \
    FAITHGUARD_MODELS_DIR=/app/models \
    # production logging
    FG_LOG_LEVEL=INFO

WORKDIR /app

# Torch-free runtime: ONNX Runtime + fastembed keep the image small and the
# container under 512 MB RAM (Render free tier).
COPY requirements.txt .
RUN pip install -r requirements.txt

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
    && mkdir -p /models-cache \
    && chown -R faithguard:faithguard /app /models-cache
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
