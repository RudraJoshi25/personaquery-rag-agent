# src/rag/cache.py
"""
Two caches:
  - embedding cache (LRU, no TTL): identical text embeds once. Embeddings are
    deterministic for a fixed model, so no expiry needed.
  - response cache (LRU + TTL): identical (mode, top_k, question) returns the
    cached answer without re-retrieving or calling the LLM.

Both expose hit/miss counters so cache behavior is provable.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


def _key(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class TTLCache:
    """Thread-safe LRU cache with optional per-entry TTL (ttl_s=0 -> no expiry)."""

    def __init__(self, maxsize: int = 256, ttl_s: float = 0.0):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, val = item
            if self.ttl_s and (time.time() - ts) > self.ttl_s:
                # expired
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return val

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), val)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._data),
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


# --- singletons -------------------------------------------------------------
_embedding_cache = TTLCache(
    maxsize=int(os.getenv("EMBED_CACHE_SIZE", "1024")),
    ttl_s=0.0,
)
_response_cache = TTLCache(
    maxsize=int(os.getenv("RESPONSE_CACHE_SIZE", "256")),
    ttl_s=float(os.getenv("RESPONSE_CACHE_TTL_S", "300")),
)


def get_embedding_cache() -> TTLCache:
    return _embedding_cache


def get_response_cache() -> TTLCache:
    return _response_cache


def cached_embed(text: str, embed_fn: Callable[[str], list[float]]) -> tuple[list[float], bool]:
    """Return (embedding, was_cache_hit). embed_fn(text) computes on miss."""
    k = _key("emb", text)
    hit = _embedding_cache.get(k)
    if hit is not None:
        return hit, True
    vec = embed_fn(text)
    _embedding_cache.set(k, vec)
    return vec, False


def response_key(question: str, mode: str, top_k: int) -> str:
    return _key("resp", mode, top_k, question.strip())
