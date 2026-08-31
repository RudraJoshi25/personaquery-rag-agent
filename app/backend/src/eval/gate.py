# src/eval/gate.py
"""
CI eval gate: retrieval-only hit-rate@k.

Runs retrieve() over the golden set and FAILS (exit 1) if retrieval
hit-rate@k drops below EVAL_HITRATE_GATE. Retrieval-only by design: no LLM
call, so it needs no GROQ key and is fully deterministic in CI.

Current measured hit-rate@8 = 100%; gate defaults to 90% to catch regressions
while tolerating minor noise.

Run:  python -m src.eval.gate
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.rag.retrieve_custom import retrieve
from src.eval.metrics import hit_rate_at_k

QA_PATH = Path(__file__).resolve().parent / "qa.jsonl"
K = int(os.getenv("EVAL_TOP_K", "8"))
GATE = float(os.getenv("EVAL_HITRATE_GATE", "0.90"))


def main() -> None:
    items = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("..."):
                items.append(json.loads(s))

    hits = 0
    for it in items:
        h = retrieve(it["question"], top_k=K)
        texts = [x["text"] for x in h]
        if hit_rate_at_k(texts, it.get("expected_snippets", []), k=K):
            hits += 1

    n = len(items)
    rate = hits / n if n else 0.0
    print(f"retrieval hit-rate@{K} = {rate*100:.1f}% ({hits}/{n}) | gate = {GATE*100:.0f}%")

    if rate < GATE:
        print(f"EVAL GATE FAILED: {rate*100:.1f}% < {GATE*100:.0f}%")
        sys.exit(1)
    print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()
