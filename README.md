# PersonaQuery — Grounded RAG over Personal Documents

PersonaQuery is a retrieval-augmented generation (RAG) assistant that answers
questions about a single person (Rudra Joshi) strictly from that person's own
documents, with inline citations and prompt-injection guardrails.

It is a **portfolio-scale** system: a small private corpus, a hybrid retriever
built on local files, and a hosted LLM. See [Scope & Limitations](#scope--limitations).

---

## Architecture

```
data/private/*.pdf + data/curated/*.md
        │ ingest (per-document chunk profiles)
        ▼
storage/  chunks.jsonl · vectors.npy · bm25.json · colbert_vecs.npz · faq_cache.json

query ─▶ guardrails ─▶ response cache ─▶ FAQ cache ─▶ query expansion
      ─▶ dense + BM25 (per variant) ─▶ RRF fusion ─▶ diversity cap
      ─▶ ColBERT rerank ─▶ context pack ─▶ Qwen3.8-27B ─▶ answer + citations
                                             (extractive fallback if the LLM is down)
```

| Layer        | Choice |
|--------------|--------|
| Backend      | FastAPI + Uvicorn (`app/backend`) |
| Frontends    | Portfolio at https://rudrajoshi.dev (React) calls the backend; `app/frontend-next` is a separate Next.js frontend |
| Embeddings   | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, no torch), 384-dim |
| Keyword      | BM25 (`rank-bm25`) |
| Fusion       | Reciprocal Rank Fusion (k=60) |
| Rerank       | ColBERT late interaction (`answerai-colbert-small-v1`), vectors precomputed at ingest |
| LLM          | `qwen/qwen3.8-27b` via Groq free tier (OpenAI-compatible), temperature 0.2 |
| Vector store | NumPy matrix + JSONL chunks; no external DB |
| Deployment   | FastAPI backend on Google Cloud Run (Docker, `app/backend/Dockerfile`) |

---

## Running locally

```bash
cd app/backend
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env                # free key from console.groq.com

python -m scripts.ingest                          # rebuild storage/ from data/
python -m scripts.precompute_faq                  # rebuild FAQ answers (calls the LLM)
python -m uvicorn src.main:app --reload           # serves /chat and /health

python -m pytest tests -q                         # unit + integration tests
python -m src.eval.gate                           # retrieval-only gate (no token needed)
python -m src.eval.run_eval                       # full eval incl. generation
```

Key environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `GROQ_API_KEY` | — | Groq API key (required for generation) |
| `GROQ_CHAT_MODEL` | `qwen/qwen3.8-27b` | answer model |
| `GROQ_EXPANSION_MODEL` | `openai/gpt-oss-20b` | query-expansion model (separate per-model rate limits) |
| `TOP_K` | 8 | retrieved chunks |
| `QUERY_EXPANSION_ENABLED` | true | LLM paraphrases for long questions |
| `RERANK_ENABLED` | true | ColBERT rerank |
| `FAQ_CACHE_ENABLED` | true | serve precomputed answers |
| `INJECTION_GUARD_ENABLED` | true | prompt-injection guard |

---

## Evaluation

`src/eval/qa.jsonl` is an 18-question golden set. CI runs the test suite and
fails if retrieval hit-rate@8 drops below 90%. Current: **94.4% (17/18)**.
The research paper and patent PDFs are deliberately not ingested (co-author
PII, reference-list noise); `data/curated/projects.md` summarises both.

### Known limitations

- **q10 ("Summarize my professional profile.") is borderline and currently
  fails retrieval.** "profile" matches the curated `linkedin-profile.md`,
  `education.md` and `projects.md` chunks about as well as the CV summary
  line, which sits just outside the top 8 (it is 3rd at top-10). It has
  flipped between pass and fail with ColBERT model and corpus changes. Left
  failing rather than tuning the golden set to pass. 18 cases is a small set,
  so treat the score as a regression alarm, not a guarantee.

---

## Scope & Limitations

- **Tiny, personal corpus** (~58 chunks). Retrieval metrics would not transfer
  unchanged to a large heterogeneous corpus.
- **Brute-force vector search** over an in-RAM matrix. Fine at this size.
- **Single LLM provider.** The FAQ cache and extractive fallback keep the bot
  answering during a provider outage; there is no second provider.
- **In-process rate limiter and caches**: single-instance only.
- **Eval is snippet/keyword based**, not a semantic judge.
- **No authentication.** Single-persona, public read-only.
