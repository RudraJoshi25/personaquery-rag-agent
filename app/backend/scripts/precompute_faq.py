#!/usr/bin/env python
"""
Precompute answers for common questions so they serve instantly at runtime.

Run AFTER ingestion, whenever the source documents change:

    python -m scripts.precompute_faq

Because this runs offline, a rate limit is a non-event: it waits and
retries instead of degrading a live visitor's answer. That's the whole
trade — move the quota risk from request time to build time.

Questions come from two places:
  - src/eval/qa.jsonl (the golden set — by definition the things worth
    getting right, and already curated)
  - FAQ_EXTRA below (phrasings a recruiter would actually type, which the
    eval set doesn't cover because it was written for measurement)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.rag.rag import run_rag  # noqa: E402
from src.rag.embedder import DEFAULT_EMBED_MODEL, get_embedder  # noqa: E402
from src.rag.faq_cache import index_sha  # noqa: E402

from src.core.config import RAG_STORAGE_DIR as STORAGE_DIR  # noqa: E402
OUT_PATH = Path(STORAGE_DIR) / "faq_cache.json"
QA_PATH = Path(__file__).resolve().parents[1] / "src" / "eval" / "qa.jsonl"

# Pacing between calls. Generous on purpose: this is a build step, nobody is
# waiting on it, and being gentle here avoids tripping provider rate limits
# mid-run.
PACE_S = float(os.getenv("FAQ_PACE_S", "6.0"))
MAX_ATTEMPTS = 3

# Real-visitor phrasings. The eval set is written for measurement ("What is
# my email address?" — first person, because it tests the identity rules);
# actual visitors ask in second/third person.
FAQ_EXTRA = [
    "How can I contact Rudra?",
    "What is Rudra's email?",
    "Where is Rudra based?",
    "Is Rudra authorised to work in Australia?",
    "What is Rudra's educational background?",
    "What did Rudra study?",
    "What are Rudra's technical skills?",
    "What programming languages does Rudra know?",
    "Does Rudra have experience with RAG or LLMs?",
    "What projects has Rudra built?",
    "Tell me about PersonaQuery.",
    "Tell me about HealthEcho.",
    "Tell me about JobPilot.",
    "What is the Cognitive-Chair project?",
    "Does Rudra have any patents?",
    "Has Rudra published any research?",
    "What awards has Rudra won?",
    "What work experience does Rudra have?",
    "What internships has Rudra done?",
    "What kind of roles is Rudra looking for?",
    "Why should we hire Rudra?",
    "Summarise Rudra's professional profile.",
    "What is Rudra's experience with Python?",
    "Does Rudra have cloud or DevOps experience?",
]


def _load_eval_questions() -> list[str]:
    if not QA_PATH.exists():
        return []
    out = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(json.loads(s)["question"])
            except Exception:
                continue
    return out


def main() -> int:
    questions = _load_eval_questions() + FAQ_EXTRA
    # De-dupe case-insensitively, preserving order.
    seen, ordered = set(), []
    for q in questions:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            ordered.append(q.strip())

    print(f"Precomputing {len(ordered)} FAQ answers (pace={PACE_S}s)\n")

    embedder = get_embedder()
    entries, failed = [], []

    for i, q in enumerate(ordered, 1):
        if i > 1:
            time.sleep(PACE_S)

        answer = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # use_faq=False: never read the cache we're in the middle of
            # rebuilding, or we'd just re-cache yesterday's answers.
            res = run_rag(q, mode="chat", use_faq=False)
            text = res.get("answer", "")
            # run_rag never raises; it degrades. Detect degraded output rather
            # than caching an apology or an extractive stopgap as if it were
            # a real answer.
            degraded = (
                "temporarily unavailable" in text.lower()
                or "having trouble generating" in text.lower()
                or not text.strip()
            )
            if not degraded:
                answer = res
                break
            wait = 20 * attempt
            print(f"  [{i}/{len(ordered)}] degraded (attempt {attempt}); waiting {wait}s")
            time.sleep(wait)

        if answer is None:
            print(f"  [{i}/{len(ordered)}] FAILED: {q}")
            failed.append(q)
            continue

        vec = np.array(embedder.embed_query(q), dtype="float32")
        vec /= (np.linalg.norm(vec) + 1e-12)

        entries.append({
            "question": q,
            "answer": answer["answer"],
            "sources": answer.get("sources", []),
            "embedding": vec.tolist(),
        })
        print(f"  [{i}/{len(ordered)}] ok: {q}")

    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embed_model": DEFAULT_EMBED_MODEL,
            "index_sha": index_sha(),
            "entries": entries,
        }, f, ensure_ascii=False)

    print(f"\nWrote {len(entries)} entries -> {OUT_PATH}")
    if failed:
        print(f"{len(failed)} question(s) could not be generated "
              f"(providers exhausted). Re-run later to fill them in:")
        for q in failed:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
