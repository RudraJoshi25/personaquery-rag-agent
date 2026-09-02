# PersonaQuery

A retrieval-augmented Q&A "digital twin" chatbot: ask questions about a person's resume, patents,
and publications and get grounded, cited answers. Backend is FastAPI + hybrid BM25/vector
retrieval + Groq for generation; frontend is Next.js.

- `app/backend` — FastAPI service (`src/`), ingestion pipeline (`scripts/ingest.py`), tests
  (`tests/`).
- `app/frontend-next` — Next.js chat UI.
- `data/private` — your source PDFs (gitignored, not in this repo).
- `app/backend/storage` — the prebuilt hybrid index (`chunks.jsonl`, `vectors.npy`, `bm25.json`),
  committed to git so the app runs without re-ingesting.

See [`docs/code-flow.md`](docs/code-flow.md) for a full trace of every request path (which
function calls which, file by file) and [`docs/flowchart.md`](docs/flowchart.md) for the visual
version of the same call graph.

## Local development

### Backend

```bash
cd app/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # then fill in GROQ_API_KEY
uvicorn src.main:app --reload
```

Or use the helper script from the repo root: `./scripts/run_local.sh`.

Run the test suite:

```bash
cd app/backend
pytest                 # full suite
pytest -m "not slow"   # skip the real-embedding-model/real-storage test
ruff check .           # lint
```

### Frontend

```bash
cd app/frontend-next
npm ci
npm run dev
```

Set `NEXT_PUBLIC_API_URL` (e.g. in `.env.local`) to point at your backend if it's not on
`http://127.0.0.1:8000`.

### Re-ingesting documents

Put PDFs/txt/md under `data/private/`, then:

```bash
cd app/backend
PYTHONPATH=. python -m scripts.ingest
```

This rebuilds `storage/{chunks.jsonl,vectors.npy,bm25.json}`. Ingestion redacts third-party emails
and phone numbers found in the source text (see `src/rag/pii_redact.py`) — spot-check the output
before committing, especially for phone numbers, since the redaction regex is intentionally
conservative to avoid mangling patent numbers/DOIs.

## Deployment

**Backend (Render):** deploys via Render's native Python buildpack, installing directly from
`app/backend/requirements.txt` — not via Docker. Required environment variables are documented in
`app/backend/.env.example`; at minimum set `GROQ_API_KEY` and `CORS_ALLOWED_ORIGINS` (the app fails
closed on CORS until the latter is set, and refuses to start entirely without the former).

For rate limiting to see real client IPs (not Render's proxy IP), add
`--proxy-headers --forwarded-allow-ips='*'` to the service's start command in Render's dashboard.

**Frontend (Vercel):** set `NEXT_PUBLIC_API_URL` to the deployed backend URL.

### Optional: Docker

`app/backend/Dockerfile` is a self-contained, tested build (multi-stage, non-root, pre-downloads
the embedding model, honors `$PORT`) for local reproducibility or future migration — it is **not**
the current production deploy path.

```bash
docker build -f app/backend/Dockerfile -t personaquery-backend app/backend
docker run -p 8000:8000 -e GROQ_API_KEY=... personaquery-backend
```
