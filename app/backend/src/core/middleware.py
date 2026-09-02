# src/core/middleware.py
from __future__ import annotations

import contextvars
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates a request id for log correlation. Must be the
    OUTERMOST middleware (registered last via app.add_middleware) so every
    response - including ones rejected by CORS or rate limiting - carries
    an X-Request-ID header."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        request_id = incoming or uuid.uuid4().hex
        # Deliberately not reset via a `finally` block: ServerErrorMiddleware
        # (which invokes the registered Exception handler for truly
        # unhandled errors) sits OUTSIDE this middleware in the stack, so a
        # reset here would erase the id as the exception unwinds through
        # this dispatch - before that handler ever runs. Each HTTP request
        # already runs in its own asyncio task under uvicorn/ASGI, so
        # leaving this set doesn't leak across requests.
        _request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """A handful of response headers appropriate for a JSON-only API served
    over HTTPS behind Render's TLS-terminating proxy. No X-Frame-Options/CSP -
    this API never returns HTML, so clickjacking/CSP headers aren't meaningful
    here."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response
