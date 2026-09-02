# PersonaQuery — Code Flow Reference

This document traces every execution path through the backend, function by function, from each
code entry point down to where it terminates. Pair it with [`flowchart.md`](./flowchart.md) for
the visual version of the same call graph. All paths are relative to `app/backend/`.

- **§1** Process startup
- **§2** Request middleware stack (runs before every route)
- **§3** `POST /chat`
- **§4** `POST /interview/start`
- **§5** `POST /interview/answer`
- **§6** `GET /health`, `/health/live`, `/health/ready`
- **§7** Shared: the retrieval layer
- **§8** Shared: the Groq generation layer
- **§9** Offline: the ingestion pipeline
- **§10** Cross-cutting concerns (where each lives)

---

## §1. Process startup — `src/main.py`

Entry point: `uvicorn src.main:app` (locally via `scripts/run_local.sh`, in the optional Docker
image via the `CMD` in `Dockerfile`, or Render's native buildpack in production).

1. `from src.core.logging import setup_logging; setup_logging()` runs **before every other
   import** in the file. `setup_logging()` (`src/core/logging.py`) configures the root Python
   logger with either `JsonFormatter` or `TextFormatter` depending on `LOG_FORMAT`, so any log
   call made during the rest of module import is already formatted correctly.
2. `validate_startup_config()` (`src/core/config.py`) runs next, **before `FastAPI()` is
   instantiated**:
   - Raises `RuntimeError` (process fails to start) if `GROQ_API_KEY` is empty, or if
     `CORS_ALLOW_ALL=true` together with `ENVIRONMENT=production` (unless
     `STRICT_STARTUP_CHECKS=false`).
   - Logs a `WARNING` (does not block startup) if `CORS_ALLOWED_ORIGINS` is empty in production,
     or if `storage/{chunks.jsonl,vectors.npy,bm25.json}` are missing — the latter is surfaced to
     callers later via `/health/ready` instead.
3. `app = FastAPI(lifespan=lifespan)` — the `lifespan` context manager (defined just above)
   logs one `"startup"` line with `vector_enabled`, `storage_ok`, and `mem_mb` (via optional
   `psutil`), then `yield`s to start serving requests.
4. Middleware is registered via repeated `app.add_middleware(...)` calls, in this order:
   `CORSMiddleware` → `SlowAPIMiddleware` → `SecurityHeadersMiddleware` → `RequestIDMiddleware`.
   Starlette treats the **last**-registered middleware as the **outermost** layer, so the actual
   per-request wrapping order (outer → inner) is `RequestIDMiddleware` →
   `SecurityHeadersMiddleware` → `SlowAPIMiddleware` → `CORSMiddleware` → the route. This is
   deliberate — see §2.
5. Four exception handlers are registered via `app.add_exception_handler(...)`:
   `RateLimitExceeded`, `StarletteHTTPException`, `RequestValidationError`, and the bare
   `Exception` catch-all. FastAPI dispatches to the most specific registered handler by exact
   type match, so these never interfere with each other.
6. `app.include_router(...)` mounts `chat_router`, `interview_router`, and `health_router`
   (from `src/api/routes_chat.py`, `routes_interview.py`, `routes_health.py`).

---

## §2. Request middleware stack — every request passes through this first

Per §1.4, incoming requests are processed outer-to-inner as:

1. **`RequestIDMiddleware`** (`src/core/middleware.py`) — reads `X-Request-ID` from the incoming
   request or generates a new UUID4 hex, stores it in a `contextvars.ContextVar`
   (`get_request_id()` reads it back anywhere in the call stack), calls the rest of the stack,
   then stamps `X-Request-ID` on the outgoing response. **Deliberately never resets the
   contextvar** in a `finally` block — see the comment in that file for why (it would erase the
   id before `Exception`-handler logging can read it, since that handler lives in
   `ServerErrorMiddleware`, outside this middleware).
2. **`SecurityHeadersMiddleware`** (same file) — adds `X-Content-Type-Options`,
   `Referrer-Policy`, `Strict-Transport-Security` to every response.
3. **`SlowAPIMiddleware`** — required by `slowapi` for the `@limiter.limit(...)` decorators used
   on individual routes (§3, §4, §5) to function; the actual limit-checking happens in those
   decorators, not here.
4. **`CORSMiddleware`** — origin allow-listing per `CORS_ALLOWED_ORIGINS` /
   `CORS_ALLOW_ORIGIN_REGEX` / `CORS_ALLOW_ALL` (see `src/core/config.py`).

If a rate limit is hit, `rate_limit_exceeded_handler` (`src/core/rate_limit.py`) builds the 429
response, computing `Retry-After` directly from the underlying `limits` library's window stats
(not via `slowapi`'s own header injection, which is incompatible with routes that return plain
dicts instead of `Response` objects — see the comment in that file).

---

## §3. `POST /chat` — `src/api/routes_chat.py::chat()`

```
chat()
 └─ run_rag()                                    src/rag/chat.py
     ├─ check_question()                          src/rag/safety/guardrails.py
     ├─ retrieve()                                 src/rag/retrieval/search.py   (§7)
     ├─ make_context_pack()                        src/rag/retrieval/search.py
     ├─ answer_with_groq()                         src/rag/generation/llm_groq.py (§8)
     └─ _scrub_output()  ×2 (answer + snippets)     src/rag/chat.py
```

1. `@limiter.limit(RATE_LIMIT_CHAT)` checks the per-IP rate limit (key: last hop of
   `X-Forwarded-For`, or the raw connection IP — see `_client_ip_key` in
   `src/core/rate_limit.py`). Exceeding it short-circuits to §2's 429 response before the
   handler body runs.
2. `ChatRequest` (Pydantic model) validates `question`: `min_length=1`, `max_length` =
   `CHAT_MAX_QUESTION_LENGTH` (default 2000), and a `field_validator` that strips whitespace and
   rejects an empty/whitespace-only result. Failing this raises `RequestValidationError`, handled
   by §1's `validation_exception_handler` → HTTP 422 with the full field-level detail.
3. `run_rag(req.question, top_k=TOP_K, mode="chat")` (`src/rag/chat.py`) is the actual
   orchestrator:
   - `check_question(question)` (`src/rag/safety/guardrails.py`) runs a regex-based
     prompt-injection filter (Unicode-normalized first, to close a homoglyph-evasion trick). If
     blocked, `run_rag` returns immediately with `{"answer": <rejection reason>, "sources": []}`
     — **no retrieval or LLM call happens** on a blocked question.
   - `retrieve(question, top_k)` (§7) returns a list of hit dicts (`text`, `metadata`, `score`).
   - Each hit is assigned a stable 1-based source id, and `make_context_pack(hits, source_ids)`
     builds the `[SOURCE n] file | p.X | section` text blocks fed to the LLM.
   - `answer_with_groq(question, context, mode="chat")` (§8) returns an `LLMResult`.
   - `_scrub_output(result.text)` applies a defensive regex email redaction (independent of the
     ingestion-time redaction in §9 — this catches anything that slipped through or predates it;
     scoped to email addresses only, gated by `DEFENSIVE_PII_REDACTION_ENABLED`).
   - `SOURCE_ID_RE` (`\[\[cite:([0-9,\s]+)\]\]`) extracts which source ids the model actually
     cited in its answer. If the model cited none, `run_rag` falls back to returning the top 3
     retrieved sources anyway (so the UI always has something to show), rather than an empty
     list.
   - Each returned source's `snippet` (first 320 chars of the chunk) is also passed through
     `_scrub_output`.
   - One structured `INFO` log line (`chat_request_completed`) is emitted with latency, retrieval
     hit count, Groq token usage, and a truncated question preview (never the full question/answer
     text — see `LOG_INCLUDE_QUESTION_PREVIEW`/`LOG_QUESTION_PREVIEW_CHARS`).
4. The response is always `200 OK` with `{"answer": str, "sources": [...]}` — even a guardrail
   rejection or a Groq failure returns 200 with an explanatory `answer` string, not an error
   status. This is a deliberate choice (see `src/rag/generation/llm_groq.py`'s `LLMResult.error`
   field): the frontend (`ChatUI.tsx`) has no special-case handling for non-200 "soft" failures,
   so degraded answers stay within the normal response shape while still being fully logged
   server-side for observability.
5. Any *unhandled* exception in this chain (e.g. the store failing to load) propagates up to
   §1's `unhandled_exception_handler` → generic 500 body with a `request_id`, never the raw
   exception text.

---

## §4. `POST /interview/start` — `src/api/routes_interview.py::interview_start()`

```
interview_start()
 └─ start_interview()                             src/rag/interview.py
     ├─ _evict_if_needed()
     ├─ _pick_seed_chunks() → get_store()           src/rag/retrieval/search.py  (§7)
     ├─ make_context_pack()                          src/rag/retrieval/search.py
     ├─ answer_with_groq(mode="interview_generate")  src/rag/generation/llm_groq.py (§8)
     └─ _parse_questions() → _fallback_questions()   src/rag/interview.py
```

1. `@limiter.limit(RATE_LIMIT_INTERVIEW_START)` — separate, stricter budget than `/chat` (default
   `5/minute;50/day`) since this call generates multiple questions in one Groq round-trip.
2. `StartReq.n_questions` is validated `3 ≤ n ≤ 12` (422 outside that range).
3. `start_interview(n_questions)`:
   - `_evict_if_needed()` runs first: lazily expires any session in the module-level `_SESSIONS`
     dict whose `last_active_at` is older than `INTERVIEW_SESSION_TTL_SECONDS`, then evicts the
     oldest sessions if the dict is at `INTERVIEW_MAX_SESSIONS` capacity. This is the only
     eviction mechanism — sessions live entirely in-process memory (a comment in the file flags
     this as a known constraint if the service is ever scaled to multiple instances/workers).
   - `_pick_seed_chunks(k=12)` calls `get_store()` (§7's public accessor) and picks 12 random
     chunks from the loaded corpus as grounding material.
   - `make_context_pack(seed_hits, max_chars=9000)` builds the context block.
   - `answer_with_groq(prompt, context, mode="interview_generate", json_mode=True)` — this mode
     selects `INTERVIEW_GENERATE_SYSTEM_PROMPT` (§8) instead of the chat system prompt, and
     `json_mode=True` sets Groq's `response_format={"type":"json_object"}`.
   - `_parse_questions(result.text, n_questions, seed_hits, had_error=result.error is not None)`:
     if the Groq call itself failed (`result.error` set), or the returned text isn't valid JSON
     matching the expected `{"questions":[{"q":..., "expected_points":[...], "anchors":[...]}]}`
     shape, it falls back to `_fallback_questions()` — which builds real, still-grounded
     questions directly from the seed chunks' `section`/`file_name` metadata (never a single
     generic canned question).
   - A new `InterviewSession` (dataclass: `session_id`, `questions`, `idx=0`, `history=[]`,
     `created_at`, `last_active_at`) is stored in `_SESSIONS[session_id]`.
4. Returns `{"session_id", "question": questions[0].q, "question_number": 1, "total": N}`.

---

## §5. `POST /interview/answer` — `src/api/routes_interview.py::interview_answer()`

```
interview_answer()
 └─ answer_interview()                             src/rag/interview.py
     └─ answer_with_groq(mode="interview_grade")     src/rag/generation/llm_groq.py (§8)
```

1. `@limiter.limit(RATE_LIMIT_INTERVIEW_ANSWER)`.
2. `AnswerReq.answer` validated (`min_length=1`, `max_length=INTERVIEW_ANSWER_MAX_LENGTH`).
3. `answer_interview(session_id, user_answer)`:
   - Looks up `_SESSIONS[session_id]`. If missing, or if `now - last_active_at >
     INTERVIEW_SESSION_TTL_SECONDS` (expired — and it's popped from the dict right there), raises
     `SessionNotFoundError`. `routes_interview.py` catches this and raises
     `HTTPException(404, ...)` — **unlike `/chat`, this path does return a real HTTP error code**,
     since the interview flow is stateful/multi-step and a silent 200 would corrupt session
     progression more confusingly than a single bad chat turn would.
   - Builds `grade_prompt` from the current question, the candidate's answer, and
     `expected_points`; `context` is the question's `anchors` (source citations captured at
     generation time in §4).
   - `answer_with_groq(grade_prompt, context, mode="interview_grade")` — this mode selects
     `INTERVIEW_GRADE_SYSTEM_PROMPT` (§8).
   - If `result.error is not None`, raises `LLMUnavailableError` → caught in
     `routes_interview.py` → `HTTPException(502, ...)`.
   - Otherwise appends `{"q", "a", "grading"}` to `session.history`, increments `session.idx`.
4. Returns either `{"done": false, "grading", "next_question", "question_number", "total"}` or,
   on the last question, `{"done": true, "grading", "summary", "history": [...]}`.

---

## §6. `GET /health`, `/health/live`, `/health/ready` — `src/api/routes_health.py`

Three routes, deliberately split by purpose (none are rate-limited):

- **`/health/live`** — always returns `{"status": "ok"}`, 200. No dependency checks at all: a
  Render/container restart wouldn't fix a missing env var or missing storage file, so this must
  never fail on those.
- **`/health/ready`** — `health_ready()` calls `_readiness_payload()`, which runs three checks:
  - `_store_loaded()` — do `storage/{chunks.jsonl,vectors.npy,bm25.json}` exist on disk.
  - `_embedder_loadable()` — has the `SentenceTransformer` model loaded successfully at least
    once (result is cached in a module-level `_embedder_loadable_cache` global — never reloads
    the model on subsequent checks). Internally calls
    `src.rag.retrieval.search._get_model()`, the same singleton loader used by real retrieval.
  - `groq_key_present` — `bool(GROQ_API_KEY)`, an env check only, **never** a live call to Groq's
    API.
  Returns `200` if all three pass, `503` otherwise, with the individual booleans in
  `checks: {...}`.
- **`/health`** — a thin alias that just calls and returns `health_ready()`, kept for backward
  compatibility with whatever health-check path an external system (e.g. Render's dashboard) may
  already be configured to hit.

---

## §7. Shared: the retrieval layer — `src/rag/retrieval/search.py`

Both `/chat` (§3) and `/interview/start` (§4) call into this module. Two module-level singletons,
lazily created and cached across requests within one process:

- `_get_store() -> HybridStore` — loads `storage/{chunks.jsonl,vectors.npy,bm25.json}` via
  `HybridStore.load()` (`src/rag/retrieval/store.py`) on first call; raises `RuntimeError` if the
  files aren't present (this is the error `/health/ready`'s `store_loaded` check is designed to
  surface before it becomes a 500). `get_store()` is a thin public wrapper around this, used by
  `src/rag/interview.py` so both the chat and interview paths share one loaded index instead of
  doubling memory usage.
- `_get_model() -> SentenceTransformer` — loads the embedding model (`EMBED_MODEL`, default
  `all-MiniLM-L6-v2`) on first call.

`retrieve(question, top_k)`:
1. `store.search_bm25(question, top_k=cand_k)` — always runs (`cand_k = max(top_k*4, 12)`, pulling
   more candidates than needed for better fusion). Internally: `HybridStore.search_bm25()` tokenizes
   the query, scores it against the `rank_bm25.BM25Okapi` index built at ingestion time, and
   returns only positive-scoring hits.
2. If `RAG_VECTOR_ENABLED`, `_get_model().encode([question])` embeds the query, then
   `store.search_vector(q_vec, top_k=cand_k)` does a cosine-similarity search against the
   normalized `vectors.npy` matrix. Wrapped in a `try/except` — a vector-search failure silently
   falls back to BM25-only results, it never fails the whole request.
3. `_rrf_fuse(vec_ranked, bm25_ranked)` combines both rankings via Reciprocal Rank Fusion
   (`1/(k+rank+1)` per list, summed per doc id, `k=60`) — chosen specifically because it needs no
   score calibration between BM25 and cosine-similarity scales.
4. Each fused hit is tagged `channel: "hybrid"|"vector"|"keyword"` depending on which of the two
   rankings it appeared in.
5. Optional `RAG_MIN_SCORE` threshold filter, then dedup by `(file_name, page_label)` keeping the
   higher-scored duplicate, then truncated to `top_k`.

`make_context_pack(hits, max_chars, source_ids)` formats hits into `[SOURCE n] file | p.X |
section\n<text>` blocks joined by `---`, stopping once `max_chars` would be exceeded (never
truncates mid-block).

---

## §8. Shared: the Groq generation layer — `src/rag/generation/llm_groq.py`

Both the chat path (§3) and both interview calls (§4, §5) go through the single function
`answer_with_groq(question, context, mode, json_mode)`, which returns an `LLMResult` dataclass
(`text`, `usage`, `error`) — never raises for a "normal" failure, so callers always get a safe
user-facing string plus a machine-readable `error` code to branch on.

1. If `GROQ_API_KEY` is empty, short-circuits with `error="misconfigured"` — no network call.
2. `SYSTEM_PROMPTS[mode]` selects the system prompt:
   - `"chat"` / `"advisor"` → `CHAT_SYSTEM_PROMPT` (grounding rules: context-only, `[[cite:N]]`
     tokens, no file names/pages in answer body, 12-word quote limit, etc.)
   - `"interview_generate"` → `INTERVIEW_GENERATE_SYSTEM_PROMPT` (raw JSON only, no citation
     tokens — deliberately different from the chat prompt, which forbids exactly what this mode
     needs to produce).
   - `"interview_grade"` → `INTERVIEW_GRADE_SYSTEM_PROMPT` (plain-text grading, no citation
     tokens needed).
3. A module-level `requests.Session()` (`_session`) with a mounted `HTTPAdapter(max_retries=Retry(...))`
   automatically retries on `429/502/503/504` (`GROQ_MAX_RETRIES`, default 2 additional attempts,
   honoring Groq's own `Retry-After` header), with a `(connect, read)` timeout tuple
   (`GROQ_TIMEOUT_CONNECT=5`, `GROQ_TIMEOUT_READ=45` by default).
4. On success (`200`): returns `LLMResult(text=<content>, usage=<groq usage dict>)`.
5. On non-200: logs the full status/body server-side, returns a generic
   `error="upstream_error"` message — the raw provider error is never sent to the client.
6. On `requests.Timeout`: `error="timeout"` with a timeout-specific user message. On any other
   `requests.RequestException`: `error="request_failed"`. Both are logged with full detail
   server-side.

---

## §9. Offline: the ingestion pipeline — not part of any live request

Entry point: `python -m scripts.ingest` (run manually, or via `./scripts/run_local.sh`'s sibling
workflow — see the root `README.md`).

```
scripts/ingest.py::main()
 ├─ _collect_paths()                                scripts/ingest.py
 └─ ingest_paths()                                  src/rag/ingestion/pipeline.py
     ├─ read_pdf_pages() / plain read()
     ├─ redact_text()                                 src/rag/ingestion/pii_redact.py
     ├─ make_chunks()                                  src/rag/ingestion/chunking.py
     ├─ embedder.embed()                               src/rag/retrieval/embedder.py
     └─ store.build() + store.save()                   src/rag/retrieval/store.py
```

1. `_collect_paths()` scans `PRIVATE_DATA_DIR` (default `data/private/`, gitignored — real source
   PDFs never live in this repo) for `.pdf`/`.txt`/`.md` files.
2. `ingest_paths(paths, store)` (`src/rag/ingestion/pipeline.py`) is the actual pipeline, capped
   at `MAX_CHUNKS` (default 600) total chunks to bound memory on low-resource hosts:
   - PDF branch: `read_pdf_pages()` (pypdf) extracts text per page; text/markdown branch just
     reads the file.
   - `redact_text(raw)` (`src/rag/ingestion/pii_redact.py`) runs **before chunking**, so
     redaction never straddles a chunk boundary: strips third-party email addresses (an
     allowlist check preserves `OWNER_EMAIL` verbatim) and phone numbers matching a
     deliberately-conservative pattern (to avoid false-positiving on patent numbers/DOIs common
     in this corpus; preserves `OWNER_PHONE`).
   - `make_chunks(text, file_name, page_label, doc_id)` (`src/rag/ingestion/chunking.py`):
     `_split_into_sections()` first splits on heading-like lines (markdown `#`, ALL-CAPS lines,
     lines ending in `:`), then `_chunk_by_words()` breaks each section into 220-word windows
     with 40-word overlap. Each chunk carries `file_name`/`page_label`/`section`/`chunk_id`
     metadata used later for citations.
   - Chunks shorter than 20 characters (pre- and post-chunking) are dropped.
3. `embedder.embed(texts)` (`src/rag/retrieval/embedder.py`) batch-encodes all chunk texts with
   `SentenceTransformer` (`EMBED_BATCH`, default 16 per batch), normalized for cosine similarity.
4. `store.build(vectors, chunks)` + `store.save()` (`src/rag/retrieval/store.py`) normalizes the
   embedding matrix, builds the `rank_bm25.BM25Okapi` index over tokenized chunk text, and writes
   `storage/chunks.jsonl` (text+metadata), `storage/vectors.npy` (float32 matrix), and
   `storage/bm25.json` (tokenized corpus) — the exact three files `_get_store()` (§7) and
   `/health/ready`'s `_store_loaded()` (§6) check for at request time.

---

## §10. Cross-cutting concerns — where each lives

| Concern | File(s) |
|---|---|
| Structured logging (`JsonFormatter`/`TextFormatter`) | `src/core/logging.py` |
| Request-ID propagation + security headers | `src/core/middleware.py` |
| Rate limiting (`Limiter`, IP key function, 429 response) | `src/core/rate_limit.py` |
| Global exception → response mapping | `src/core/errors.py` |
| All environment variables + startup fail-fast checks | `src/core/config.py` |
| Env var documentation | `app/backend/.env.example` |
| Prompt-injection filtering | `src/rag/safety/guardrails.py` |
| Third-party PII redaction (ingestion-time) | `src/rag/ingestion/pii_redact.py` |
| Third-party PII redaction (defensive, output-side) | `src/rag/chat.py::_scrub_output` |
