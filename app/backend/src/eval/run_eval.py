# src/eval/run_eval.py
"""
PersonaQuery evaluation harness.

For each QA item in qa.jsonl it measures three independent signals:

  1. retrieval : hit@k and recall@k of the expected source snippets, computed
                 against the raw top-k chunks from retrieve().
  2. answer    : keyword match of expected substrings in the generated answer.
  3. citation  : presence of an inline [[cite:...]] token in the answer.

Aggregate numbers are printed to stdout AND saved to a timestamped JSON file
under src/eval/results/ so latency-free eval history is retained per run.

Run from app/backend/:  python -m src.eval.run_eval
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.core.config import RAG_STORAGE_DIR as STORAGE_DIR
from src.rag.rag import run_rag
from src.rag.retrieve_custom import retrieve
from src.eval.metrics import (
    has_citation,
    keyword_match,
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
)

QA_PATH = Path(__file__).resolve().parent / "qa.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K = int(os.getenv("EVAL_TOP_K", "8"))

# Pace calls so 20 back-to-back questions (each ~3K tokens, sometimes doubled
# by query expansion) don't trip the provider's rate limit.
EVAL_PACE_S = float(os.getenv("EVAL_PACE_S", "3.0"))

# The eval bypasses the precomputed FAQ cache by default.
#
# Every question in qa.jsonl is also seeded into the FAQ cache (the cache is
# built FROM this file), so leaving the cache on would make the eval grade
# its own cached output instead of the live pipeline — it would report a
# healthy score even if retrieval or generation had regressed badly, because
# it would never touch either. Set EVAL_USE_FAQ=true only to measure what a
# real visitor gets end-to-end, cache included.
EVAL_USE_FAQ = os.getenv("EVAL_USE_FAQ", "false").lower() in {"1", "true", "yes"}


def _load_qa() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("..."):
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return items


def _warn_if_index_stale() -> None:
    """Shout if the index is older than the code or data that produced it.

    Two full eval runs were wasted measuring a stale index: ingestion crashed
    (an out-of-memory abort in the embedder), the shell ran the eval anyway,
    and the numbers looked like a code change had simply failed to help. The
    eval cannot tell a bad change from an unapplied one, so it has to say when
    it might be measuring the past.
    """
    idx = Path(STORAGE_DIR) / "chunks.jsonl"
    if not idx.exists():
        print("!! storage/chunks.jsonl missing — run: python -m scripts.ingest\n")
        return

    idx_mtime = idx.stat().st_mtime
    roots = [Path("src/rag"), Path("../../data/private"), Path("../../data/curated")]
    newer = [
        f for root in roots if root.exists()
        for f in root.rglob("*")
        if f.is_file() and f.stat().st_mtime > idx_mtime
    ]
    if newer:
        names = ", ".join(sorted({f.name for f in newer})[:5])
        print(
            f"!! WARNING: {len(newer)} file(s) are NEWER than the index "
            f"({names}{'…' if len(newer) > 5 else ''}).\n"
            f"!! You are probably measuring a STALE index. "
            f"Re-run: python -m scripts.ingest\n"
        )


def run() -> Dict[str, Any]:
    _warn_if_index_stale()
    qa = _load_qa()
    per_item: List[Dict[str, Any]] = []

    n = len(qa)
    retr_hits = 0
    recall_sum = 0.0
    precision_sum = 0.0
    answer_pass = 0
    citation_pass = 0

    print(f"Running eval on {n} QA pairs (top_k={TOP_K}, "
          f"faq_cache={'ON' if EVAL_USE_FAQ else 'OFF (live pipeline)'})\n")

    for i, item in enumerate(qa):
        if i > 0 and EVAL_PACE_S > 0:
            time.sleep(EVAL_PACE_S)

        q = item["question"]

        # --- retrieval metric: raw top-k chunks ---
        hits = retrieve(q, top_k=TOP_K)
        retrieved_texts = [h["text"] for h in hits]
        retrieved_files = [h["metadata"].get("file_name", "") for h in hits]
        expected_snippets = item.get("expected_snippets", [])
        expected_source = item.get("expected_source", "")

        hit = hit_rate_at_k(retrieved_texts, expected_snippets, k=TOP_K)
        rec = recall_at_k(retrieved_texts, expected_snippets, k=TOP_K)
        prec = precision_at_k(retrieved_files, expected_source, k=TOP_K)

        # --- answer + citation metrics: full RAG generation ---
        res = run_rag(q, top_k=TOP_K, mode="chat", use_faq=EVAL_USE_FAQ)
        ans = res.get("answer", "")

        kw_ok, missing = keyword_match(ans, item.get("must_include", []))
        cite_ok = has_citation(ans) if item.get("must_cite", True) else True

        retr_hits += 1 if hit else 0
        recall_sum += rec
        precision_sum += prec
        answer_pass += 1 if kw_ok else 0
        citation_pass += 1 if cite_ok else 0

        status = "PASS" if (hit and kw_ok and cite_ok) else "FAIL"
        print(
            f"{item['id']} {status:4} | hit@{TOP_K}={int(hit)} "
            f"recall={rec:.2f} kw={'Y' if kw_ok else 'N'} "
            f"cite={'Y' if cite_ok else 'N'} | {q[:48]}"
        )
        if not kw_ok:
            print(f"      missing keywords: {missing}")

        per_item.append({
            "id": item["id"],
            "question": q,
            "hit_at_k": bool(hit),
            "recall_at_k": round(rec, 4),
            "source_precision_at_k": round(prec, 4),
            "answer_keyword_pass": bool(kw_ok),
            "missing_keywords": missing,
            "citation_pass": bool(cite_ok),
            "answer": ans,
            "retrieved_files": retrieved_files,
        })

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": n,
        "top_k": TOP_K,
        "retrieval_hit_rate": round(retr_hits / n, 4) if n else 0.0,
        "retrieval_mean_recall_at_k": round(recall_sum / n, 4) if n else 0.0,
        "retrieval_mean_source_precision_at_k": round(precision_sum / n, 4) if n else 0.0,
        "answer_pass_rate": round(answer_pass / n, 4) if n else 0.0,
        "citation_pass_rate": round(citation_pass / n, 4) if n else 0.0,
    }

    print("\n" + "=" * 46)
    print(f"  N                          : {summary['n']}")
    print(f"  Retrieval hit-rate @{TOP_K}      : {summary['retrieval_hit_rate']*100:.1f}%  ({retr_hits}/{n})")
    print(f"  Retrieval mean recall@{TOP_K}    : {summary['retrieval_mean_recall_at_k']*100:.1f}%")
    print(f"  Retrieval mean src prec@{TOP_K}  : {summary['retrieval_mean_source_precision_at_k']*100:.1f}%")
    print(f"  Answer keyword pass-rate   : {summary['answer_pass_rate']*100:.1f}%  ({answer_pass}/{n})")
    print(f"  Citation pass-rate         : {summary['citation_pass_rate']*100:.1f}%  ({citation_pass}/{n})")
    print("=" * 46)

    # persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": per_item}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results -> {out_path}")

    return summary


if __name__ == "__main__":
    run()
