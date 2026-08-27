"""Central configuration for FaithGuard.

Everything is overridable via environment variables so the same code runs in
local dev, pytest, Docker, and cloud deploys without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FAITHGUARD_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("FAITHGUARD_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("FAITHGUARD_REPORTS_DIR", PROJECT_ROOT / "reports"))


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size: int = _env_int("FG_CHUNK_SIZE", 800)          # chars per chunk
    chunk_overlap: int = _env_int("FG_CHUNK_OVERLAP", 120)
    top_k: int = _env_int("FG_TOP_K", 5)                       # passages into prompt
    re_top_k: int = _env_int("FG_RE_TOP_K", 8)                 # passages after re-retrieval
    rrf_k: int = 60                                            # Reciprocal Rank Fusion constant
    bm25_weight: float = _env_float("FG_BM25_WEIGHT", 1.0)
    dense_weight: float = _env_float("FG_DENSE_WEIGHT", 1.0)
    embedding_model: str = _env("FG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


@dataclass(frozen=True)
class DetectionConfig:
    nli_model: str = _env("FG_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    # Answer-level verdict thresholds (applied to claim-level outputs)
    unsupported_claim_threshold: float = _env_float("FG_UNSUPPORTED_T", 0.5)
    hallucinated_answer_fraction: float = _env_float("FG_HALLU_FRACTION", 0.35)
    # A claim whose best NLI score is below this counts as "weakly supported"
    min_claim_chars: int = 15
    batch_size: int = _env_int("FG_NLI_BATCH", 32)


@dataclass(frozen=True)
class MitigationConfig:
    max_rounds: int = _env_int("FG_MAX_MITIGATION_ROUNDS", 2)
    abstain_text: str = (
        "I could not verify an answer from the available sources. "
        "The retrieved context does not contain sufficient supported information."
    )


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = _env("FG_LLM_PROVIDER", "openai_compatible")
    base_url: str = _env("FG_LLM_BASE_URL", "https://api.tokenrouter.com/v1")
    api_key_env: str = _env("FG_LLM_KEY_ENV", "HERMES_CUSTOM_TOKENROUTER_API_KEY")
    model: str = _env("FG_LLM_MODEL", "qwen/qwen3.8-max-free")
    temperature: float = _env_float("FG_LLM_TEMP", 0.2)
    max_tokens: int = _env_int("FG_LLM_MAX_TOKENS", 512)
    timeout_s: float = _env_float("FG_LLM_TIMEOUT", 90.0)
    max_retries: int = _env_int("FG_LLM_RETRIES", 4)


@dataclass(frozen=True)
class Settings:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: str = _env("FG_DEVICE", "cpu")
    log_level: str = _env("FG_LOG_LEVEL", "INFO")

    @property
    def llm_api_key(self) -> str | None:
        return os.environ.get(self.generation.api_key_env)


def get_settings() -> Settings:
    return Settings()
