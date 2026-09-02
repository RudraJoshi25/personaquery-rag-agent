# PersonaQuery — System Flowchart

Visual companion to [`code-flow.md`](./code-flow.md), which explains every box below in prose with
exact function/file references. Read this diagram top-down: process startup, then the middleware
every request passes through, then each of the four route groups, then the two shared layers
(retrieval, Groq generation) that the routes call into, and finally the offline ingestion pipeline
that populates the data those routes read from.

```mermaid
flowchart TD
    subgraph START["Process startup — src/main.py"]
        A1["setup_logging()<br/>src/core/logging.py<br/>(must run before any other import)"] --> A2["validate_startup_config()<br/>src/core/config.py<br/>fail-fast: no GROQ_API_KEY ⇒ process exits"]
        A2 --> A3["FastAPI(lifespan=lifespan)<br/>lifespan logs storage_ok / vector_enabled / mem_mb"]
        A3 --> A4["Middleware registered (innermost→outermost):<br/>CORS → SlowAPI → SecurityHeaders → RequestID"]
        A4 --> A5["Exception handlers registered:<br/>RateLimitExceeded, HTTPException, RequestValidationError, Exception"]
        A5 --> A6["Routers mounted:<br/>chat_router, interview_router, health_router"]
    end

    A6 --> REQ(["Incoming HTTP request"])
    REQ --> MW["Middleware stack (outer→inner):<br/>RequestIDMiddleware → SecurityHeadersMiddleware →<br/>SlowAPIMiddleware → CORSMiddleware<br/>src/core/middleware.py"]
    MW --> ROUTE{"Which route?"}

    %% ---------------- CHAT ----------------
    ROUTE -->|"POST /chat"| C1["chat()<br/>src/api/routes_chat.py"]
    C1 --> C2["@limiter.limit(RATE_LIMIT_CHAT)<br/>src/core/rate_limit.py — 429 if exceeded"]
    C2 --> C3["ChatRequest validated<br/>(1–2000 chars, stripped) — 422 if invalid"]
    C3 --> C4["run_rag(question, top_k, mode='chat')<br/>src/rag/chat.py"]
    C4 --> C5["check_question()<br/>src/rag/safety/guardrails.py"]
    C5 -->|"blocked"| C6(["{answer: rejection reason, sources: []}"])
    C5 -->|"allowed"| C7["retrieve(question, top_k)"]
    C7 -.-> RET
    RET -.-> C8["make_context_pack(hits)<br/>builds [SOURCE n] file | p.X | section blocks"]
    C8 --> C9["answer_with_groq(question, context, mode='chat')"]
    C9 -.-> GROQ
    GROQ -.-> C10["parse [[cite:N]] tokens from answer;<br/>fallback to top-3 sources if none cited"]
    C10 --> C11["_scrub_output()<br/>defensive regex email redaction"]
    C11 --> C12(["{answer, sources[]}"])

    %% ---------------- INTERVIEW START ----------------
    ROUTE -->|"POST /interview/start"| I1["interview_start()<br/>src/api/routes_interview.py"]
    I1 --> I2["@limiter.limit(RATE_LIMIT_INTERVIEW_START)"]
    I2 --> I3["start_interview(n_questions)<br/>src/rag/interview.py"]
    I3 --> I4["_evict_if_needed()<br/>TTL sweep + max-session cap on _SESSIONS"]
    I4 --> I5["_pick_seed_chunks() → get_store()"]
    I5 -.-> RET
    RET -.-> I6["make_context_pack(seed_hits, max_chars=9000)"]
    I6 --> I7["answer_with_groq(prompt, context,<br/>mode='interview_generate', json_mode=True)"]
    I7 -.-> GROQ
    GROQ -.-> I8{"result.error?"}
    I8 -->|"yes / bad JSON"| I9["_fallback_questions()<br/>grounded questions built from seed-chunk metadata"]
    I8 -->|"no"| I10["_parse_questions() parses JSON schema"]
    I9 --> I11["new InterviewSession stored in _SESSIONS[session_id]"]
    I10 --> I11
    I11 --> I12(["{session_id, question, question_number, total}"])

    %% ---------------- INTERVIEW ANSWER ----------------
    ROUTE -->|"POST /interview/answer"| J1["interview_answer()<br/>src/api/routes_interview.py"]
    J1 --> J2["@limiter.limit(RATE_LIMIT_INTERVIEW_ANSWER)"]
    J2 --> J3["answer_interview(session_id, answer)<br/>src/rag/interview.py"]
    J3 -->|"unknown / TTL-expired"| J4["raise SessionNotFoundError"] --> J4b(["HTTP 404"])
    J3 -->|"found"| J5["answer_with_groq(grade_prompt, anchors,<br/>mode='interview_grade')"]
    J5 -.-> GROQ
    GROQ -.-> J6{"result.error?"}
    J6 -->|"yes"| J7["raise LLMUnavailableError"] --> J7b(["HTTP 502"])
    J6 -->|"no"| J8["append {q, a, grading} to session.history; idx += 1"]
    J8 --> J9(["{done, grading, next_question | summary+history}"])

    %% ---------------- HEALTH ----------------
    ROUTE -->|"GET /health/live"| H1(["{status: ok} — always 200, no dependency checks"])
    ROUTE -->|"GET /health/ready"| H2["health_ready()<br/>src/api/routes_health.py"]
    H2 --> H3["_store_loaded() +  _embedder_loadable() (cached) +<br/>groq_key_present (env check only)"]
    H3 --> H4(["200 if all pass, else 503"])
    ROUTE -->|"GET /health"| H5["health() → delegates to health_ready()<br/>(back-compat alias)"]

    %% ---------------- RETRIEVAL LAYER ----------------
    subgraph RET["Retrieval layer — src/rag/retrieval/search.py"]
        direction TB
        R1["_get_store() singleton<br/>HybridStore.load() from storage/*"] --> R2["store.search_bm25(query)<br/>rank_bm25 BM25Okapi"]
        R1 --> R3["_get_model() singleton<br/>SentenceTransformer.encode(query)"]
        R3 --> R4["store.search_vector(query_vec)<br/>cosine similarity"]
        R2 --> R5["_rrf_fuse()<br/>Reciprocal Rank Fusion of both rankings"]
        R4 --> R5
        R5 --> R6["dedupe by (file_name, page_label),<br/>sort desc by score, truncate to top_k"]
    end

    %% ---------------- GROQ LAYER ----------------
    subgraph GROQ["Generation layer — src/rag/generation/llm_groq.py"]
        direction TB
        G1["answer_with_groq(question, context, mode, json_mode)"] --> G2["SYSTEM_PROMPTS[mode] selects system prompt<br/>(chat/advisor vs interview_generate vs interview_grade)"]
        G2 --> G3["_session.post() to Groq API<br/>retry on 429/502/503/504, timeout=(connect,read)"]
        G3 -->|"200"| G4(["LLMResult(text, usage, error=None)"])
        G3 -->|"non-200 / timeout / exception"| G5(["LLMResult(text=generic user message, error=code)"])
    end

    %% ---------------- INGESTION (offline) ----------------
    subgraph ING["Offline — python -m scripts.ingest (not part of a live request)"]
        direction TB
        P1["scripts/ingest.py main()"] --> P2["_collect_paths()<br/>scans data/private/**/*.{pdf,txt,md}"]
        P2 --> P3["ingest_paths(paths, store)<br/>src/rag/ingestion/pipeline.py"]
        P3 --> P4["read_pdf_pages() (pypdf) or plain read() for txt/md"]
        P4 --> P5["redact_text()<br/>src/rag/ingestion/pii_redact.py<br/>strips 3rd-party emails/phones, keeps OWNER_EMAIL/PHONE"]
        P5 --> P6["make_chunks()<br/>src/rag/ingestion/chunking.py<br/>heading-aware section split + 220-word windows, 40-word overlap"]
        P6 --> P7["embedder.embed(texts)<br/>src/rag/retrieval/embedder.py<br/>SentenceTransformer batch encode"]
        P7 --> P8["store.build(vectors, chunks) + store.save()<br/>src/rag/retrieval/store.py<br/>→ storage/{chunks.jsonl, vectors.npy, bm25.json}"]
    end

    P8 -.->|"read at request time by"| R1
```

## Legend

| Shape | Meaning |
|---|---|
| Rounded rect `(["..."])` | Terminal response returned to the HTTP client |
| Diamond `{"..."}` | Branch point |
| Solid arrow | Direct function call |
| Dotted arrow | Call into / return from one of the two shared subgraphs (retrieval, generation), or the ingestion pipeline handing off to storage |
