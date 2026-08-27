"""Production observability: request IDs, structured logging, metrics.

Stdlib-only. Exposes a tiny in-process metrics registry (counters + latency
histograms) surfaced at ``GET /metrics`` in a Prometheus-ish text format.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("faithguard.observability")

REQUEST_ID_HEADER = "x-request-id"


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach/propagate an x-request-id and stash it for log correlation."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One structured log line per request with latency + request id."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rid = getattr(request.state, "request_id", "-")
        log.info(
            "method=%s path=%s status=%d latency_ms=%.1f rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            rid,
        )
        # feed the metrics registry
        metrics.observe_request(request.url.path, response.status_code, elapsed_ms)
        return response


class Metrics:
    """Minimal thread-safe metrics registry (counters + latency)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests_total = 0
        self.by_status: dict[int, int] = defaultdict(int)
        self.by_path: dict[str, int] = defaultdict(int)
        self.latency_sum_ms = 0.0
        self.latency_max_ms = 0.0
        self.started_at = time.time()
        # domain counters
        self.ask_total = 0
        self.ask_mitigated = 0
        self.ask_abstained = 0
        self.detect_total = 0
        self.detect_flagged = 0

    def observe_request(self, path: str, status_code: int, latency_ms: float):
        with self._lock:
            self.requests_total += 1
            self.by_status[status_code] += 1
            self.by_path[path] += 1
            self.latency_sum_ms += latency_ms
            self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    def inc(self, name: str, n: int = 1):
        with self._lock:
            setattr(self, name, getattr(self, name, 0) + n)

    def snapshot(self) -> dict:
        with self._lock:
            avg = (self.latency_sum_ms / self.requests_total) if self.requests_total else 0.0
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "requests_total": self.requests_total,
                "by_status": dict(self.by_status),
                "by_path": dict(self.by_path),
                "latency_avg_ms": round(avg, 1),
                "latency_max_ms": round(self.latency_max_ms, 1),
                "ask_total": self.ask_total,
                "ask_mitigated": self.ask_mitigated,
                "ask_abstained": self.ask_abstained,
                "detect_total": self.detect_total,
                "detect_flagged": self.detect_flagged,
            }

    def prometheus_text(self) -> str:
        s = self.snapshot()
        lines = [
            "# HELP faithguard_requests_total Total HTTP requests",
            "# TYPE faithguard_requests_total counter",
            f"faithguard_requests_total {s['requests_total']}",
            "# HELP faithguard_latency_avg_ms Average request latency",
            "# TYPE faithguard_latency_avg_ms gauge",
            f"faithguard_latency_avg_ms {s['latency_avg_ms']}",
            "# HELP faithguard_ask_total Total /ask calls",
            "# TYPE faithguard_ask_total counter",
            f"faithguard_ask_total {s['ask_total']}",
            "# HELP faithguard_ask_mitigated_total /ask calls that triggered mitigation",
            "# TYPE faithguard_ask_mitigated_total counter",
            f"faithguard_ask_mitigated_total {s['ask_mitigated']}",
            "# HELP faithguard_ask_abstained_total /ask calls that abstained",
            "# TYPE faithguard_ask_abstained_total counter",
            f"faithguard_ask_abstained_total {s['ask_abstained']}",
            "# HELP faithguard_detect_total Total /detect calls",
            "# TYPE faithguard_detect_total counter",
            f"faithguard_detect_total {s['detect_total']}",
            "# HELP faithguard_detect_flagged_total /detect calls flagged hallucinated",
            "# TYPE faithguard_detect_flagged_total counter",
            f"faithguard_detect_flagged_total {s['detect_flagged']}",
        ]
        return "\n".join(lines) + "\n"


# module-level singleton
metrics = Metrics()
