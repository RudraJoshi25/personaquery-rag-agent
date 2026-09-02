# src/core/rate_limit.py
from __future__ import annotations

import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from src.core.middleware import get_request_id

logger = logging.getLogger("personaquery.rate_limit")


def _client_ip_key(request: Request) -> str:
    """Render sits in front of this app as a reverse proxy and sets
    X-Forwarded-For. Take the LAST entry - a client can prepend arbitrary
    fake IPs to this header, but cannot control what Render itself appends
    as the final hop. Falls back to the raw connection IP if the header is
    absent (e.g. local dev)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


# In-memory storage: correct for a single Render instance. If this ever
# scales to multiple instances, swap storage_uri to a shared backend, e.g.
# storage_uri="redis://<host>:6379" - no other code changes needed, `limits`
# (which slowapi wraps) abstracts the backend.
# headers_enabled=False (default): slowapi's own header injection tries to
# treat every decorated endpoint's return value as a Starlette Response,
# which breaks FastAPI routes that return plain dicts. Retry-After is added
# manually below, scoped to just the 429 path, instead.
limiter = Limiter(key_func=_client_ip_key, storage_uri="memory://")


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning(
        "rate_limited",
        extra={"path": request.url.path, "client_ip": _client_ip_key(request), "limit": str(exc.detail)},
    )
    response = JSONResponse(
        status_code=429,
        content={
            "error": "rate_limited",
            "message": "Too many requests. Please slow down and try again.",
            "request_id": get_request_id(),
        },
    )

    # Compute Retry-After directly from the underlying `limits` storage
    # rather than via slowapi's _inject_headers (which assumes every
    # decorated endpoint returns a Response - not true for FastAPI routes
    # that return plain dicts, so calling it there breaks success responses).
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if view_rate_limit is not None:
        try:
            rate_limit_item, identifiers = view_rate_limit
            window_stats = limiter.limiter.get_window_stats(rate_limit_item, *identifiers)
            retry_after = max(1, int(window_stats.reset_time - time.time()))
            response.headers["Retry-After"] = str(retry_after)
        except Exception:
            logger.debug("could_not_compute_retry_after", exc_info=True)

    return response
