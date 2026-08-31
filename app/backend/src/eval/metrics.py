# src/eval/metrics.py
"""
Evaluation metric functions for the PersonaQuery RAG harness.

Three independent signals, all computed from real run output:

  retrieval : hit_rate@k / recall@k   -> did the correct source chunk(s)
                                          appear in the top-k retrieved chunks?
  answer    : keyword_match            -> does the answer contain the expected
                                          substrings drawn from the source docs?
  citation  : has_citation             -> did the answer emit an inline citation
                                          token in the required format?

Kept dependency-free (stdlib only) so the harness stays reproducible.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

# Inline citation token produced by the LLM, e.g. [[cite:1,3]]. Also accepts
# full-width bracket variants (e.g. 【cite:1,3】): some models substitute
# typographic punctuation despite being told to use plain ASCII, and a
# malformed-but-recognizable citation shouldn't fail the eval when the intent
# is clearly there.
CITE_RE = re.compile(r"[\[【]{1,2}\s*cite:\s*[0-9]+(?:\s*,\s*[0-9]+)*\s*[\]】]{1,2}")

# Citation anchor used inside context packs, e.g. [file.pdf | p.1 | SECTION].
ANCHOR_RE = re.compile(r"\[[^\]]+\|\s*p\.[^\]]+\|\s*[^\]]+\]")

# Reduce text to bare alphanumerics only: no spaces, no hyphens, no dashes
# (of any Unicode variant), no dots, no punctuation of any kind. This is
# deliberately NOT a list of specific characters to normalize — enumerating
# separators (this dash, that dash, a space-in-digits rule...) is a losing
# game, since the model can always reformat with a punctuation mark we didn't
# anticipate. Reducing to "the alphanumeric content, in order" sidesteps the
# question of *which* separator was used entirely: "HC-06", "HC 06", "HC.06",
# and "HC‑06" (non-breaking hyphen) all become "hc06" and compare equal.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace, for citation-anchor style matching
    where word boundaries still matter."""
    return " ".join((text or "").lower().split())


def _squash(text: str) -> str:
    """Alphanumeric-only, order-preserving fingerprint of the text. Used for
    fact matching (keywords, snippets) so formatting differences in
    separators never cause a false miss."""
    return _NON_ALNUM_RE.sub("", (text or "").lower())


def has_citation(answer: str) -> bool:
    """True if the answer contains an inline [[cite:...]] token or a full anchor."""
    a = answer or ""
    return bool(CITE_RE.search(a) or ANCHOR_RE.search(a))


def keyword_match(answer: str, must_include: Sequence[str]) -> Tuple[bool, List[str]]:
    """
    Returns (all_present, missing). An empty must_include trivially passes.
    Matching is on alphanumeric content only (see _squash) so it can't be
    defeated by reformatting a fact with different separator characters.
    """
    a = _squash(answer)
    missing = [kw for kw in (must_include or []) if _squash(kw) not in a]
    return (len(missing) == 0, missing)


def _found_snippets(
    retrieved_texts: Iterable[str],
    expected_snippets: Sequence[str],
    k: int,
) -> List[bool]:
    """For each expected snippet, whether it appears in any of the top-k chunks."""
    # "|" (non-alphanumeric, stripped out of every squashed snippet) keeps
    # chunks from silently fusing at their boundaries — without it, a snippet
    # that never actually appears could false-positive by matching the tail
    # of one chunk concatenated with the head of the next.
    top = [_squash(t) for t in list(retrieved_texts)[:k]]
    blob = "|".join(top)
    return [_squash(s) in blob for s in (expected_snippets or [])]


def hit_rate_at_k(
    retrieved_texts: Iterable[str],
    expected_snippets: Sequence[str],
    k: int = 8,
) -> bool:
    """
    Hit@k: True if AT LEAST ONE expected snippet appears in the top-k chunks.
    If no expected snippets are given, returns True (nothing to retrieve).
    """
    if not expected_snippets:
        return True
    return any(_found_snippets(retrieved_texts, expected_snippets, k))


def recall_at_k(
    retrieved_texts: Iterable[str],
    expected_snippets: Sequence[str],
    k: int = 8,
) -> float:
    """
    Recall@k: fraction of expected snippets present in the top-k chunks.
    Returns 1.0 when no snippets are specified.
    """
    if not expected_snippets:
        return 1.0
    found = _found_snippets(retrieved_texts, expected_snippets, k)
    return sum(1 for f in found if f) / len(found)


def precision_at_k(
    retrieved_files: Iterable[str],
    expected_source: str,
    k: int = 8,
) -> float:
    """
    Source precision@k: fraction of top-k chunks that come from the expected
    source document. 0.0 when expected_source is falsy.
    """
    if not expected_source:
        return 0.0
    top = list(retrieved_files)[:k]
    if not top:
        return 0.0
    hits = sum(1 for f in top if _norm(f) == _norm(expected_source))
    return hits / len(top)
