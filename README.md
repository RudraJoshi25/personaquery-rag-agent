# PersonaQuery — Grounded RAG over Personal Documents

PersonaQuery is a retrieval-augmented generation (RAG) assistant that answers
questions about a single person (Rudra Joshi) strictly from that person's own
documents — a resume, a research paper, and a patent — with inline citations
and prompt-injection guardrails.

It is a **portfolio-scale** system: a small private corpus, a hybrid retriever
built on local files, and a hosted LLM. It is engineered to be correct,
measurable, and reproducible rather than to handle enterprise load. See
[Scope & Limitations](#scope--limitations).

---

## Architecture

```
PDFs ──ingest──▶ chunks.jsonl + vectors.npy + bm25.json (storage/)
                          │
   query ──▶ guardrails ──▶ hybrid retrieve ──▶ context pack ──▶ Groq LLM ──▶ answer + citations
                            (BM25 + vector,        (cited)        (Llama 3.3)
                             RRF fusion,
                             identity boost)
```

| Layer        | Choice |
|--------------|--------|
| Backend      | FastAPI + Uvicorn (`app/backend`) |
| Frontend     | Next.js (`app/frontend-next`), deployable to Vercel |
| Embeddings   | `all-MiniLM-L6-v2` (sentence-transformers), 384-dim, cosine |
| Keyword      | BM25 (`rank-bm25`, Okapi) |
| Fusion       | Reciprocal Rank Fusion (RRF, k=60) |
| LLM          | `llama-3.3-70b-versatile` via Groq API, temperature 0.2 |
| Vector store | NumPy matrix (`vectors.npy`) + JSONL chunks; no external DB |
| Deployment   | Backend on Render, frontend on Vercel |

---

## Measured metrics

All numbers below are produced by code in this repo, not estimates.

### Corpus
| Metric | Value | Source |
|--------|-------|--------|
| Source documents | 3 PDFs (resume, research paper, patent) | `data/private/` |
| Chunks indexed | 84 | `storage/chunks.jsonl` |
| Per-document | resume 8, patent 7, paper 69 | `storage/chunks.jsonl` |
| Chunk size / overlap | 220 words / 40 words, section-aware | `src/rag/chunking.py` |

### Retrieval & answer quality (20-question golden set)
Run with `python -m src.eval.run_eval` (top_k = 8):

| Metric | Value |
|--------|-------|
| Retrieval hit-rate@8 | **100%** (20/20) |
| Retrieval mean recall@8 | **100%** |
| Retrieval mean source-precision@8 | **65%** |
| Answer keyword pass-rate | **95%** (19/20) |
| Citation pass-rate | **100%** (20/20) |

The single answer miss is a paraphrase ("HealthEcho" vs "HealthEchoLink"); the
correct chunk was retrieved (recall 1.0). Results are saved per run to
`src/eval/results/eval_<timestamp>.json`.

### Latency
Run with `python latency_bench.py`; every live query also appends to
`logs/latency.jsonl` with a rolling p50/p95/mean summary in
`logs/latency_summary.json` (`src/core/latency.py`).

| Stage | p50 | p95 | mean | n |
|-------|-----|-----|------|---|
| Local hybrid retrieval (embed + BM25 + vector + RRF) | 37 ms | 85 ms | 39 ms | 20 |
| End-to-end (retrieval + Groq LLM) | ~2.0 s | ~2–3 s* | — | 20 |

Local retrieval is fully reproducible and provider-independent (measured above).
End-to-end latency is dominated by the Groq LLM call (~1.4–2 s); p50 ≈ 2.0 s
measured over 20 hybrid queries.

\* Groq's **free tier** rate-limits bursty traffic. The client retries with
backoff, so a throttled request can spike to 100 s+. Those spikes are a provider
limit, not system latency, and are excluded from the typical p95 above.

---

## Guardrails

| Guardrail | Where |
|-----------|-------|
| Prompt-injection detection (10 regex patterns) | `src/rag/guardrails.py` → `check_question()` |
| Empty-question + control-char sanitization | `src/rag/guardrails.py` |
| Request schema validation (Pydantic) | `src/api/routes_chat.py` (`ChatRequest`) |
| Citation enforcement (every paragraph must cite) | `src/rag/llm_groq.py` system prompt |
| Cite-only-if-used (strip uncited sources) | `src/rag/rag.py` |
| Identity disambiguation (user vs. co-authors) | `src/rag/llm_groq.py` rule 11 + `retrieve_custom.py` identity boost |
| Context-size cap (12k chars) | `src/rag/retrieve_custom.py` → `make_context_pack()` |

### Identity disambiguation (fixed defect)
The research paper and patent list co-authors (Kshitij Joshi, Pujan Soni,
Abhilasha Vyas, Smit Joshi, Priyadarshi, Dr Kanta Prasad Sharma). A query like
"what is my email" previously surfaced a co-author's contact details because
(a) vector retrieval was disabled in deployment, (b) the paper supplies 69 of
84 chunks, and (c) there was no source weighting. The fix:

- re-enabled hybrid (vector + BM25) retrieval,
- added an **identity-query detector** that boosts resume chunks for
  first-person/contact questions (`src/rag/retrieve_custom.py`),
- added a **system-prompt rule** that maps "I/me/my/Rudra" to the resume owner
  and forbids presenting co-author contact details as the user's.

Result: "what is my email" now returns `rudrajoshi.cs@gmail.com` citing the
resume only.

---

## Running locally

```bash
# Backend (from app/backend)
pip install -r requirements.txt
# set GROQ_API_KEY in .env
python -m uvicorn src.main:app --reload         # serves /chat and /health

# (Re)build the index from data/private/*.pdf
python scripts/ingest.py

# Evaluate
python -m src.eval.run_eval                      # writes src/eval/results/

# Latency benchmark
python latency_bench.py                          # writes logs/latency*.json
```

Key environment variables (`app/backend/.env`):

| Var | Default | Purpose |
|-----|---------|---------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | embedding model |
| `TOP_K` | 8 | retrieved chunks |
| `RAG_VECTOR_ENABLED` | true | enable vector half of hybrid retrieval |
| `INJECTION_GUARD_ENABLED` | true | enable prompt-injection guard |

---

## Scope & Limitations

This project is deliberately scoped to portfolio demonstration, not production
scale. Honest constraints:

- **Corpus is tiny and personal.** 3 documents / 84 chunks. Retrieval metrics
  (100% hit-rate@8) reflect a small, well-separated corpus and would not
  transfer unchanged to a large heterogeneous one.
- **Vector store is an in-memory NumPy matrix**, not a real vector DB (no ANN
  index, no sharding). Fine at 84 vectors; would not scale to millions.
- **Single hosted LLM on a free tier.** Groq's free tier rate-limits bursty
  traffic; the client retries with backoff, so a throttled request can take
  tens of seconds. Reported latency is the unthrottled typical case.
- **Memory-constrained deployment.** Vector retrieval was previously disabled on
  the Render free tier to save RAM; it is re-enabled here, which raises memory
  use. On very small instances, set `RAG_VECTOR_ENABLED=false` to fall back to
  BM25-only (with reduced semantic recall).
- **Eval is keyword/snippet based**, not a semantic judge. It verifies the right
  source was retrieved and expected facts appear, but does not score fluency or
  catch subtle hallucination beyond the citation requirement.
- **No authentication / multi-tenant isolation.** Single-persona, single-user.
