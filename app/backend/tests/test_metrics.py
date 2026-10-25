# tests/test_metrics.py
"""Phase 6: eval metric functions."""
from __future__ import annotations

import pytest

from src.eval.metrics import (
    has_citation,
    hit_rate_at_k,
    keyword_match,
    precision_at_k,
    recall_at_k,
)


def test_has_citation():
    assert has_citation("answer [[cite:1,2]]")
    assert has_citation("see [file.pdf | p.1 | SECTION]")
    assert not has_citation("no citation here")


def test_keyword_match():
    ok, missing = keyword_match("Hello World", ["hello"])
    assert ok and missing == []
    ok, missing = keyword_match("Hello", ["bye"])
    assert not ok and missing == ["bye"]
    ok, _ = keyword_match("anything", [])      # nothing required -> pass
    assert ok


def test_hit_rate_at_k_and_k_boundary():
    texts = ["miss one", "miss two", "the target value"]
    assert hit_rate_at_k(texts, ["target value"], k=3)
    assert not hit_rate_at_k(texts, ["target value"], k=2)   # target beyond top-2
    assert not hit_rate_at_k(texts, ["nope"], k=3)
    assert hit_rate_at_k(texts, [], k=3)                     # no expectation -> True


def test_recall_at_k():
    texts = ["aaa bbb", "ccc"]
    assert recall_at_k(texts, ["aaa", "ccc"], k=2) == 1.0
    assert recall_at_k(texts, ["aaa", "zzz"], k=2) == 0.5
    assert recall_at_k(texts, [], k=2) == 1.0


def test_precision_at_k():
    files = ["resume.pdf", "paper.pdf", "resume.pdf"]
    assert precision_at_k(files, "resume.pdf", k=3) == pytest.approx(2 / 3)
    assert precision_at_k(files, "", k=3) == 0.0
