# FaithGuard — Hallucination Detection & Mitigation Engine
# CPU-only image. Models are downloaded at first startup and cached in /models-cache.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models-cache \
    FAITHGUARD_DATA_DIR=/app/data \
    FAITHGUARD_MODELS_DIR=/app/models

WORKDIR /app

# CPU torch first (small wheel), then the rest
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY faithguard ./faithguard
COPY scripts ./scripts
COPY pyproject.toml .

# Optional: bake the benchmark corpus + trained classifier into the image
COPY data/corpus.json* ./data/
COPY models/*.pkl* ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=5)" || exit 1

CMD ["uvicorn", "faithguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
