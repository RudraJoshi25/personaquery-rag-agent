# src/eval/run_eval.py
from __future__ import annotations
import json
import re
from src.rag.rag import run_rag

CITE_RE = re.compile(r"\[[^\]]+\|\s*p\.[^\]]+\|\s*[^\]]+\]")

def run():
    passed = 0
    total = 0
    with open("src/eval/qa.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if line.strip().startswith("#") or line.strip().startswith("..."):
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            total += 1
            res = run_rag(item["question"], top_k=12, mode="chat")
            ans = res["answer"]

            ok = True
            for s in item.get("must_include", []):
                if s.lower() not in ans.lower():
                    ok = False

            if item.get("must_cite", True) and not CITE_RE.search(ans):
                ok = False

            print(f"{item['id']} -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                print("Question:", item["question"])
                print("Answer:", ans[:500], "\n")

            passed += 1 if ok else 0

    print(f"\nScore: {passed}/{total}")

if __name__ == "__main__":
    run()
