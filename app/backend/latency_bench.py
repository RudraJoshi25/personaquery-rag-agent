"""
Latency benchmark: 20 queries, reports p50, p95, mean end-to-end.
Run from app/backend/: python latency_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

SAMPLE_QUERIES = [
    "Who is Rudra Joshi?",
    "What are Rudra's skills?",
    "What projects has Rudra worked on?",
    "What is the Cognitive-Chair project?",
    "What patent is described in the documents?",
    "What is Rudra's educational background?",
    "What programming languages does Rudra know?",
    "Describe the brain-computer interface work.",
    "What AI/ML frameworks has Rudra used?",
    "What is Rudra's work experience?",
    "What is the purpose of the patent described?",
    "What awards or achievements does Rudra have?",
    "What is Rudra's email or contact info?",
    "Describe the wheelchair project in detail.",
    "What machine learning models are mentioned?",
    "What is Rudra's role in the Cognitive-Chair project?",
    "What technologies are used in the patent?",
    "What soft skills does Rudra have?",
    "What internships or jobs has Rudra held?",
    "Summarize all documents in one paragraph.",
]

def main():
    from src.rag.rag import run_rag

    latencies: list[float] = []
    print(f"Running {len(SAMPLE_QUERIES)} queries...\n")

    for i, q in enumerate(SAMPLE_QUERIES, 1):
        t0 = time.perf_counter()
        try:
            run_rag(q, top_k=8, mode="chat")
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  [{i:02d}] ERROR: {e}")
            continue
        latencies.append(elapsed)
        print(f"  [{i:02d}] {elapsed:.2f}s  '{q[:50]}'")
        time.sleep(4.0)  # space calls to stay under provider rate limits

    if not latencies:
        print("No successful queries.")
        return

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = statistics.median(latencies_sorted)
    p95_idx = min(int(0.95 * n), n - 1)
    p95 = latencies_sorted[p95_idx]
    mean = statistics.mean(latencies_sorted)

    print(f"\n{'='*40}")
    print(f"  N queries : {n}")
    print(f"  Mean      : {mean:.2f}s")
    print(f"  p50       : {p50:.2f}s")
    print(f"  p95       : {p95:.2f}s")
    print(f"  Min       : {latencies_sorted[0]:.2f}s")
    print(f"  Max       : {latencies_sorted[-1]:.2f}s")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
