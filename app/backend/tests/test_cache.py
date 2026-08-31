# tests/test_cache.py
"""Phase 6: cache behavior — TTL, LRU eviction, embedding reuse."""
from __future__ import annotations

import time

from src.rag.cache import TTLCache, cached_embed, get_embedding_cache


def test_hit_and_miss_counters():
    c = TTLCache(maxsize=10, ttl_s=0)
    assert c.get("k") is None          # miss
    c.set("k", "v")
    assert c.get("k") == "v"           # hit
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1


def test_ttl_expiry():
    c = TTLCache(maxsize=10, ttl_s=0.05)
    c.set("k", "v")
    assert c.get("k") == "v"
    time.sleep(0.08)
    assert c.get("k") is None          # expired


def test_lru_eviction():
    c = TTLCache(maxsize=2, ttl_s=0)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)                      # evicts oldest "a"
    assert c.get("a") is None
    assert c.get("b") == 2 and c.get("c") == 3


def test_cached_embed_reuses_and_counts():
    get_embedding_cache().clear()
    calls = {"n": 0}

    def emb(_t):
        calls["n"] += 1
        return [1.0, 2.0, 3.0]

    v1, hit1 = cached_embed("same text", emb)
    v2, hit2 = cached_embed("same text", emb)
    assert hit1 is False and hit2 is True
    assert calls["n"] == 1             # embedded once
    assert v1 == v2
