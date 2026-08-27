"""Production security: API-key auth + in-memory rate limiting.

Design goals
------------
- Zero new dependencies (stdlib only) so the image stays small.
- Auth is OFF when ``FG_API_KEY`` is unset (frictionless local dev) but the
  app logs a loud warning so nobody ships an open endpoint by accident.
- Rate limiting is a per-key sliding window. Cheap, good enough for a
  single-instance CPU service. Swap for Redis if you scale out.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

log = logging.getLogger("faithguard.security")

# Header the client sends the key in.
AUTH_HEADER = "x-api-key"
BEARER_PREFIX = "bearer "


def configured_api_key() -> str | None:
    """The API key the server expects, or None if auth is disabled."""
    key = os.environ.get("FG_API_KEY", "").strip()
    return key or None


def auth_enabled() -> bool:
    return configured_api_key() is not None


def _extract_key(request: Request) -> str | None:
    """Pull the key from x-api-key header or an Authorization: Bearer header."""
    key = request.headers.get(AUTH_HEADER)
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith(BEARER_PREFIX):
        return auth[len(BEARER_PREFIX):].strip()
    return None


def require_api_key(request: Request) -> None:
    """FastAPI dependency. Raises 401 when auth is on and the key is wrong.

    Uses ``secrets.compare_digest`` to avoid timing side-channels.
    """
    expected = configured_api_key()
    if expected is None:
        return  # auth disabled (dev)
    provided = _extract_key(request)
    # compare_digest on str raises TypeError for non-ASCII — compare UTF-8
    # bytes instead (still constant-time, works for any key content).
    if not provided or not secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class SlidingWindowRateLimiter:
    """Thread-safe per-key sliding-window rate limiter.

    ``limit`` requests per ``window_s`` seconds, tracked per key. Memory is
    bounded: stale keys are pruned lazily on access.
    """

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            # opportunistic prune of dead keys to bound memory
            if len(self._hits) > 1024:
                dead = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
                for k in dead:
                    self._hits.pop(k, None)
            return True

    def retry_after(self, key: str) -> float:
        with self._lock:
            dq = self._hits.get(key)
            if not dq:
                return 0.0
            return max(0.0, dq[0] + self.window_s - time.monotonic())


def build_rate_limiter() -> SlidingWindowRateLimiter | None:
    """Create a limiter from env, or None if limiting is disabled."""
    limit = int(os.environ.get("FG_RATE_LIMIT", "0") or 0)   # 0 = off
    window = float(os.environ.get("FG_RATE_WINDOW_S", "60") or 60)
    if limit <= 0:
        return None
    return SlidingWindowRateLimiter(limit=limit, window_s=window)


def enforce_rate_limit(limiter: SlidingWindowRateLimiter | None, request: Request) -> None:
    """Raise 429 when the caller has exceeded their window."""
    if limiter is None:
        return
    key = _extract_key(request)
    if not key:
        key = request.client.host if request.client else "anonymous"
    if not limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(limiter.retry_after(key)) + 1)},
        )
