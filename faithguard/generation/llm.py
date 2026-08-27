"""OpenAI-compatible chat client with retries and backoff.

Works with any OpenAI-compatible endpoint: OpenRouter, Tokenrouter, vLLM,
Ollama, LM Studio, Azure OpenAI, etc. Uses plain `requests` — no SDK pinning.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import requests

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    attempts: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 90.0,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session = requests.Session()

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        g = settings.generation
        return cls(
            base_url=g.base_url,
            api_key=os.environ.get(g.api_key_env),
            model=g.model,
            temperature=g.temperature,
            max_tokens=g.max_tokens,
            timeout_s=g.timeout_s,
            max_retries=g.max_retries,
        )

    def chat(self, system: str, user: str, temperature: float | None = None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }

        last_err = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                    wait = max(retry_after, 2.0 * attempt)
                    log.warning("LLM 429 rate-limited; sleeping %.1fs (attempt %d)", wait, attempt)
                    time.sleep(min(wait, 30.0))
                    last_err = "rate_limited"
                    continue
                if resp.status_code >= 500:
                    last_err = f"server_error_{resp.status_code}"
                    # exponential backoff for overloaded free-tier endpoints
                    wait = min(2.0 ** attempt + 1.0, 45.0)
                    log.warning("LLM %d server error; sleeping %.1fs (attempt %d)",
                                resp.status_code, wait, attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                if not text:
                    # Reasoning models can burn the whole token budget on
                    # reasoning and return empty content — retryable.
                    last_err = "empty_content"
                    log.warning("LLM returned empty content (attempt %d); retrying", attempt)
                    time.sleep(1.0 * attempt)
                    continue
                return LLMResponse(text=text, model=data.get("model", self.model),
                                   usage=data.get("usage", {}), attempts=attempt)
            except requests.RequestException as e:
                last_err = f"request_error: {e}"
                log.warning("LLM request failed (attempt %d): %s", attempt, e)
                time.sleep(1.5 * attempt)
            except (KeyError, IndexError, ValueError) as e:
                last_err = f"bad_response: {e}"
                break
        return LLMResponse(text="", model=self.model, attempts=self.max_retries, error=last_err)
