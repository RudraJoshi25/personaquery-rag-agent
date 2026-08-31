# src/core/latency.py
"""
Persistent latency logging.

Every RAG query appends one line to logs/latency.jsonl and the rolling
p50/p95/mean summary is recomputed into logs/latency_summary.json, so there
is an ongoing recorded latency history rather than one-off benchmark output.

Disable with RAG_LATENCY_LOG=0.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "logs"))
LATENCY_LOG = LOG_DIR / "latency.jsonl"
LATENCY_SUMMARY = LOG_DIR / "latency_summary.json"

_ENABLED = os.getenv("RAG_LATENCY_LOG", "1").lower() in {"1", "true", "yes"}


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile on a non-empty list."""
    s = sorted(values)
    idx = min(int(pct * len(s)), len(s) - 1)
    return s[idx]


def record_latency(
    total_s: float,
    retrieve_s: float | None = None,
    llm_s: float | None = None,
    mode: str = "chat",
    ok: bool = True,
) -> None:
    """Append one latency record and refresh the rolling summary."""
    if not _ENABLED:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "total_s": round(total_s, 4),
            "retrieve_s": round(retrieve_s, 4) if retrieve_s is not None else None,
            "llm_s": round(llm_s, 4) if llm_s is not None else None,
            "mode": mode,
            "ok": ok,
        }
        with open(LATENCY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        _refresh_summary()
    except Exception:
        # Never let latency logging break a request.
        pass


def _refresh_summary() -> Dict[str, Any]:
    totals = [r["total_s"] for r in _read_log() if r.get("total_s") is not None]
    if not totals:
        return {}
    summary = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "n": len(totals),
        "mean_s": round(statistics.mean(totals), 4),
        "p50_s": round(_percentile(totals, 0.50), 4),
        "p95_s": round(_percentile(totals, 0.95), 4),
        "min_s": round(min(totals), 4),
        "max_s": round(max(totals), 4),
    }
    with open(LATENCY_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _read_log() -> List[Dict[str, Any]]:
    if not LATENCY_LOG.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(LATENCY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def get_summary() -> Dict[str, Any]:
    """Return the current rolling latency summary (recomputed from the log)."""
    return _refresh_summary()


if __name__ == "__main__":
    print(json.dumps(get_summary(), indent=2))
