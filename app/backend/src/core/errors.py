# src/core/errors.py
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.middleware import get_request_id

logger = logging.getLogger("personaquery.errors")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        extra={"path": request.url.path, "exc_type": type(exc).__name__},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong. Please try again.",
            "request_id": get_request_id(),
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(
        "http_exception",
        extra={"path": request.url.path, "status_code": exc.status_code, "detail": exc.detail},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "request_id": get_request_id(),
        },
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    logger.info("validation_error", extra={"path": request.url.path, "errors": errors})
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": errors,
            "request_id": get_request_id(),
        },
    )
