# src/api/rate_limit.py
"""
In-process per-client rate limiter (sliding window).

SINGLE-INSTANCE ONLY. State lives in this process's memory and does NOT
coordinate across replicas. A multi-replica production deployment MUST replace
this with a shared store (e.g. Redis INCR + EXPIRE, or a token-bucket in
Redis), otherwise each replica enforces the limit independently and a client
could send up to limit * N_replicas. This is intentional for portfolio /
single-instance scale — do not mistake it for distributed rate limiting.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from src.core.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_S, TRUST_PROXY
from src.core.logging import log_event
from src.rag.cache import TTLCache


class InProcessRateLimiter:
    def __init__(self, limit: int, window_s: float,
                 maxsize: int = 10_000, ttl_s: float = 120.0):
        self.limit = limit
        self.window_s = window_s
        # Bounded: an attacker rotating IPs can't grow memory without limit.
        # ttl_s must exceed window_s, or an active client's history expires
        # mid-window. LRU eviction under a flood only ever forgets a client
        # (a fresh window), never blocks one.
        self._hits = TTLCache(maxsize=maxsize, ttl_s=max(ttl_s, window_s))
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_s). Records the hit when allowed."""
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            dq = self._hits.get(client)
            if dq is None:
                dq = deque()
            while dq and dq[0] < cutoff:
                dq.popleft()
            self._hits.set(client, dq)  # refreshes TTL + LRU position
            if len(dq) >= self.limit:
                retry_after = self.window_s - (now - dq[0])
                return False, max(0.0, retry_after)
            dq.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter: InProcessRateLimiter | None = None


def get_limiter() -> InProcessRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = InProcessRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_S)
    return _limiter


def client_id(request: Request) -> str:
    # The FIRST X-Forwarded-For hop is whatever the client sent, so trusting
    # it let anyone mint a fresh rate-limit identity per request. The LAST
    # hop is appended by our own proxy and can't be forged — but only when a
    # proxy is actually there, hence opt-in (render.yaml turns it on).
    # ponytail: assumes exactly one trusted proxy hop; count hops from the
    # right if a CDN is ever put in front of Render.
    xff = request.headers.get("x-forwarded-for")
    # TEMPORARY: confirm Render adds exactly one XFF hop (the last entry
    # should be the caller's real IP). Logs IPs: remove once verified.
    log_event("INFO", "xff_debug", xff=xff,
              peer=request.client.host if request.client else None)
    if TRUST_PROXY and xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_dep(request: Request) -> str:
    """FastAPI dependency: throttle per client, raise a clean 429 with Retry-After."""
    limiter = get_limiter()
    cid = client_id(request)
    allowed, retry_after = limiter.check(cid)
    if not allowed:
        log_event("WARN", "rate_limited", client=cid, retry_after_s=round(retry_after, 1))
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and try again shortly.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return cid
