# src/core/logging.py
"""
Structured JSON logging: one JSON object per line, queryable.

Every event is written to stdout (for container log capture) and appended to
logs/app.jsonl. Use a trace_id to correlate all events for one request.

    log_event("INFO", "retrieval", trace_id=tid, hits=8, latency_ms=37)
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from src.core.config import LOG_DIR
APP_LOG = LOG_DIR / "app.jsonl"

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}
_MIN_LEVEL = _LEVELS.get(os.getenv("LOG_LEVEL", "INFO").upper(), 20)
_TO_FILE = os.getenv("LOG_TO_FILE", "1").lower() in {"1", "true", "yes"}
_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "1").lower() in {"1", "true", "yes"}

_lock = threading.Lock()


def log_event(level: str, event: str, *, trace_id: str | None = None, **fields: Any) -> dict:
    """Emit one structured log record. Returns the record (handy for tests)."""
    level = level.upper()
    rec: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": level,
        "event": event,
    }
    if trace_id:
        rec["trace_id"] = trace_id
    rec.update(fields)

    if _LEVELS.get(level, 20) < _MIN_LEVEL:
        return rec

    line = json.dumps(rec, default=str, ensure_ascii=False)
    try:
        if _TO_STDOUT:
            stream = sys.stderr if level == "ERROR" else sys.stdout
            print(line, file=stream, flush=True)
        if _TO_FILE:
            with _lock:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                with open(APP_LOG, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
    except Exception:
        pass  # logging must never break the request
    return rec


def new_trace_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
