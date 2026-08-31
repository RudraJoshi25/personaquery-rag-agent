# src/rag/resilience.py
"""
LLM resilience primitives: typed errors, retry-with-backoff+jitter, and a
thread-safe circuit breaker. Provider-agnostic — used to wrap Groq calls.
"""
from __future__ import annotations

import os
import random
import threading
import time
from typing import Callable, Optional, Tuple, Type, TypeVar


# --- typed errors -----------------------------------------------------------
class LLMError(Exception):
    """Base LLM failure. `code` is a stable, caller-safe identifier.

    `code` can be overridden per-instance so a caller can attach the actual
    HTTP status (e.g. "http_400"). Without this every non-retryable failure
    logged as the generic "llm_error", which made a 400 indistinguishable
    from a 401 or 404 in production logs — the status was put in the message
    and then thrown away by the logging path.
    """
    code = "llm_error"

    def __init__(self, message: str = "", code: Optional[str] = None):
        super().__init__(message)
        if code:
            self.code = code


class LLMRateLimited(LLMError):
    code = "rate_limited"

    def __init__(self, message: str = "", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    code = "timeout"


class LLMServerError(LLMError):
    code = "server_error"


class CircuitOpen(LLMError):
    code = "circuit_open"


# Transient failures worth retrying.
RETRYABLE: Tuple[Type[Exception], ...] = (LLMRateLimited, LLMTimeout, LLMServerError)

T = TypeVar("T")


# --- retry ------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable: Tuple[Type[Exception], ...] = RETRYABLE,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Call fn(); on a retryable exception, retry with exponential backoff
    (base_delay * 2**attempt) plus optional full jitter, capped at max_delay.
    Honors exc.retry_after when present. Re-raises after max_retries.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except retryable as e:
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            if jitter:
                delay += random.uniform(0.0, base_delay)
            retry_after = getattr(e, "retry_after", None)
            if retry_after:
                delay = min(max_delay, max(delay, float(retry_after)))
            if on_retry:
                on_retry(attempt, e, delay)
            sleep(delay)
            attempt += 1


# --- circuit breaker --------------------------------------------------------
class CircuitBreaker:
    """
    States: closed -> (failures >= threshold) -> open -> (cooldown) ->
    half_open -> (success) -> closed | (failure) -> open.
    """

    def __init__(self, fail_threshold: int = 5, cooldown_s: float = 30.0,
                 now: Callable[[], float] = time.monotonic):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._now = now
        self._lock = threading.Lock()
        self.state = "closed"
        self.fails = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        """Whether a call may proceed. Transitions open->half_open after cooldown."""
        with self._lock:
            if self.state == "open":
                if self._now() - self.opened_at >= self.cooldown_s:
                    self.state = "half_open"
                    return True
                return False
            return True  # closed or half_open (single trial)

    def record_success(self) -> None:
        with self._lock:
            self.state = "closed"
            self.fails = 0

    def record_failure(self) -> None:
        with self._lock:
            self.fails += 1
            if self.state == "half_open" or self.fails >= self.fail_threshold:
                self.state = "open"
                self.opened_at = self._now()

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "fails": self.fails,
                    "fail_threshold": self.fail_threshold, "cooldown_s": self.cooldown_s}

    def reset(self) -> None:
        with self._lock:
            self.state = "closed"
            self.fails = 0
            self.opened_at = 0.0


# Process-wide breaker for the LLM provider.
_breaker: Optional[CircuitBreaker] = None


def get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker(
            fail_threshold=int(os.getenv("CB_FAIL_THRESHOLD", "5")),
            cooldown_s=float(os.getenv("CB_COOLDOWN_S", "30")),
        )
    return _breaker
