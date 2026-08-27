"""Circuit breaker + retry policy for external LLM calls.

Fail-fast on repeated upstream failures so the API doesn't hang and
callers get quick 503s instead of long timeouts.
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum

log = logging.getLogger("faithguard.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"      # normal, requests pass through
    OPEN = "open"          # failing, fast-fail
    HALF_OPEN = "half_open"  # testing if upstream recovered


class CircuitBreaker:
    """Simple thread-safe circuit breaker for a single upstream."""

    def __init__(
        self,
        failure_threshold: int = 5,      # consecutive failures to open
        success_threshold: int = 2,      # consecutive successes to close
        timeout_s: float = 30.0,         # time in OPEN before HALF_OPEN
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_s = timeout_s

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._opened_at and (time.monotonic() - self._opened_at) >= self.timeout_s:
                    log.info("Circuit breaker entering HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._successes = 0
            return self._state

    def call(self, func, *args, **kwargs):
        """Execute func with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker open; failing fast")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self.success_threshold:
                    log.info("Circuit breaker CLOSED after recovery")
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    self._successes = 0
            elif self._state == CircuitState.CLOSED:
                self._failures = 0

    def _on_failure(self):
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # any failure in half-open reopens immediately
                log.warning("Circuit breaker re-opened from HALF_OPEN")
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._failures = 0
                self._successes = 0
            elif self._state == CircuitState.CLOSED:
                if self._failures >= self.failure_threshold:
                    log.warning(
                        "Circuit breaker OPENED after %d consecutive failures",
                        self._failures,
                    )
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()

    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0
            self._opened_at = None


class CircuitOpenError(Exception):
    pass