# Claude Code Audit Prompt — PersonaQuery RAG

Paste everything below the line into Claude Code from the repo root
(`personaquery-rag-agent/`). It is written to be self-contained: it tells the
agent what the system is, what has already been tried, what the constraints
are, what to inspect, and exactly what shape of answer is acceptable.

---

## ROLE

You are a senior RAG / information-retrieval engineer doing a full technical
audit of a production-facing retrieval-augmented generation service. Your job
is **not** to make the eval score go up. Your job is to find the real reasons
this system retrieves or answers imperfectly, and to propose changes that
would help an arbitrary unseen user question — ranked by
(expected accuracy gain) ÷ (implementation risk + latency cost).

Be skeptical. Verify claims against the code and the data before asserting
them. If you cannot verify something in this environment, say so explicitly
rather than guessing.

---

## SITUATION

### What the product is

PersonaQuery is a grounded RAG assistant that answers questions about one
person (Rudra Joshi) strictly from his own documents — résumé, an IEEE
research paper, a patent filing, and hand-curated markdown. It is embedded as
an "Ask AI" widget in a public portfolio site, so the primary audience is
**recruiters asking unpredictable, casually-phrased questions**. Wrong or
ungrounded answers are worse than "I don't know".

### Repository layout

```
app/backend/
  src/rag/
    prompt.py            canonical system prompt + user-prompt builder
    llm_anthropic.py     sole generation provider (Anthropic Messages API)
    retrieve_custom.py   the retrieval pipeline (entry: retrieve())
    query_expansion.py   conditional multi-query expansion
    rerank.py            ColBERT late-interaction reranking
    embedder.py          FastEmbed (ONNX) dense embeddings
    chunking.py          section-aware chunking
    doc_profiles.py      per-document-type chunk profiles + diversity cap
    store.py             HybridStore: chunks.jsonl + vectors.npy + bm25.json
    vector_backend.py    Chroma backend (swappable)
    faq_cache.py         precomputed-answer layer (semantic match)
    cache.py             embedding cache + response TTL cache
    resilience.py        retry w/ backoff + circuit breaker
    guardrails.py        prompt-injection screening
    rag.py               orchestration: guardrails → cache → FAQ → retrieve → LLM
  src/eval/
    qa.jsonl             20-item golden set
    run_eval.py          harness
    metrics.py           hit@k, recall@k, source-precision@k, keyword, citation
    gate.py              CI regression gate
  scripts/
    ingest.py            build the index
    precompute_faq.py    build the FAQ cache
  tests/                 pytest suite
data/private/            3 source PDFs (résumé, patent, IEEE paper)
data/curated/            4 hand-written .md files
```

### Current pipeline, end to end

1. **Guardrails** — prompt-injection screening, length limits, rate limiting.
2. **Response cache** — in-memory LRU + TTL, keyed on (question, mode, top_k).
3. **FAQ cache** — 44 precomputed answers; a visitor question is embedded and
   cosine-matched against precomputed question embeddings; threshold 0.86; a
   hit skips retrieval **and** generation entirely (~10 ms, zero API cost).
4. **Query expansion** — conditional. Short/simple factual lookups skip it;
   otherwise Claude Haiku generates 3 paraphrases. Costs ~1.5–2 s when it runs.
5. **Hybrid retrieval per query variant** — BM25 (rank-bm25 Okapi) over chunk
   text, plus dense cosine over FastEmbed `BAAI/bge-small-en-v1.5` (384-dim,
   aliased from `all-MiniLM-L6-v2`). Variants run in a thread pool.
6. **Reciprocal Rank Fusion** across all ranked lists, k=60.
7. **Source-diversity cap** — no single file may take more than 4 of the
   returned slots; overflow is appended, not dropped.
8. **Content-keyed dedup**, then optional score threshold.
9. **ColBERT late-interaction rerank** (`colbert-ir/colbertv2.0` via FastEmbed
   ONNX). Document token vectors are **precomputed at ingest** and stored in
   `storage/colbert_vecs.npz`; only the query is encoded at request time.
10. **Generation** — Anthropic Messages API, `claude-sonnet-5`,
    `max_tokens=1024`, temperature **omitted** (the model rejects it).
11. **Extractive fallback** — if generation fails entirely, return the top
    retrieved source text verbatim with citations rather than an error.

### Measured state (most recent full run, live pipeline, FAQ cache bypassed)

```
N                          : 20
Retrieval hit-rate @8      : 85–90%
Retrieval mean recall@8    : 85–90%
Retrieval mean src prec@8  : ~36%      <-- lowest number in the suite
Answer keyword pass-rate   : 90–95%
Citation pass-rate         : 95–100%
```

Latency, uncached: retrieval 0.1–2.4 s (query expansion dominates when it
fires), generation 2.5–11 s, total ~3–11 s. FAQ-cache hits are ~10 ms.

Corpus: ~92 chunks after the latest ingest.

### Bugs ALREADY found and fixed — do not re-report these as new

Read these; they tell you the failure modes this codebase actually has.

1. **Chunk imbalance.** One 20-page paper produced 82% of the index; identity
   questions returned co-authors' contact details. Fixed with per-document
   chunk profiles + a source-agnostic diversity cap (replacing a regex
   "identity boost" hack).
2. **ColBERT re-encoding per query.** Documents were re-encoded on every
   request (60–100 s/query on CPU). Fixed by precomputing document vectors at
   ingest.
3. **Editorial HTML comments were indexed.** `<!-- ... -->` notes in the
   curated markdown became chunks. The note explaining that a phone number was
   *deliberately excluded* outranked the résumé chunk containing the number, so
   the bot answered "not stated". Comments are now stripped for `.md`.
4. **PDF icon-glyph corruption.** The LaTeX résumé's FontAwesome icons survive
   extraction fused to values: the email tokenized as `perudrajoshi`, not
   `rudrajoshi`, so BM25 could never match it. Now normalized into clean
   fields (`Email: …`, `Phone: …`).
5. **A regression from fixing (4).** The first version *deleted* icon labels,
   which removed the only occurrence of the word "phone" from the résumé and
   dropped hit-rate 90%→85%. Labels are now rewritten, not deleted.
6. **Stale vectors on corpus shrink.** Chroma is keyed by positional chunk
   index; upsert never deleted ids past the end of a shrunken corpus, so
   retrieval returned an out-of-range id and `IndexError`-ed **every** query.
   Fixed with collection reset at ingest + an out-of-range guard.
7. **A fact hardcoded into the system prompt.** The canonical email was
   written into the prompt; the model recited it while stating it was "not in
   the context" — right answer, broken grounding. Removed.
8. **Provider churn.** Groq deprecated its models mid-project; a four-provider
   failover chain (Groq/Cerebras/NVIDIA) existed only to work around free-tier
   token caps and was deleted when generation moved to prepaid Anthropic.

### Known-open issues

- **`src prec@8` ≈ 36%** — most retrieved chunks come from files other than the
  one the question is "about". This may be fine (diversity is deliberate) or
  may indicate the ranking is too flat. Investigate before assuming either.
- **Two eval questions fail persistently**, both vocabulary mismatches:
  - `"Summarize my professional profile."` — the résumé SUMMARY chunk contains
    neither "professional" nor "profile" (BM25 rank ~39).
  - `"What is my contact email (not a co-author's)?"` — the phrase
    "co-author" pulls the paper/awards chunks to the top (target rank ~44).
- **Latency is 3–11 s uncached**, against an original target of 1–2 s.
- **The FAQ cache and the eval set overlap**, so `run_eval` bypasses the cache
  by default (`EVAL_USE_FAQ=false`) to avoid grading the cache.

### Hard constraints

- **Single provider: Anthropic.** Do not propose re-adding Groq/Cerebras/NVIDIA.
- **Deployment: Render free tier, Docker, ~512 MB.** No GPU. No torch — the
  embedding stack is ONNX/FastEmbed deliberately. Heavy dependencies are a
  real cost, not a detail.
- **Corpus is tiny (~92 chunks) and static between ingests.** Solutions that
  only pay off at 10k+ chunks are not automatically wins here; say so if you
  propose one.
- **Answers must remain grounded and cited.** No facts in the system prompt.
- **Do not optimise against the 20 eval questions.** See rules below.

---

## WHAT TO AUDIT

Work through these. For each, look at the actual code and, where possible, the
actual data in `storage/` and `data/`.

### A. Retrieval quality
- Is BM25 + dense + RRF the right fusion here? Is `RRF_K=60` sensible for a
  92-chunk corpus, or is it flattening genuine score separation?
- Is `top_k=8` right? What does recall@k look like as k varies?
- Is the diversity cap (4/source) helping or capping out the correct document?
  Quantify against the golden set if you can.
- Does ColBERT reranking actually improve ordering here, or is it neutral
  noise at this corpus size? Measure with it on vs off.
- Is `src prec@8 ≈ 36%` a problem or an artifact of the metric? Argue from data.

### B. Chunking and document representation
- Inspect `chunking.py` + `doc_profiles.py` against the real chunks in
  `storage/chunks.jsonl`. Are boundaries sane? Any orphaned or truncated
  chunks? Any chunk that is pure noise?
- Do chunks carry enough context to be independently interpretable? Consider
  contextual chunk headers (document title/type prepended) — a general
  technique, not a fix for two specific questions. Assess cost/benefit.
- Is anything in the corpus duplicated across résumé / LinkedIn / projects
  markdown in a way that splits retrieval signal?

### C. Query understanding
- Is the conditional expansion heuristic (`needs_expansion`) well-calibrated?
  It skips short factual lookups — verify against real question shapes.
- Is 3 variants right? Does expansion measurably improve recall, and is it
  worth its ~1.5–2 s?
- Would HyDE, entity extraction, or a query classifier help more than
  paraphrase expansion, given a tiny static corpus?

### D. Generation and grounding
- Review `prompt.py`. Is anything in it a *fact* rather than an *instruction*?
  (Facts belong in documents.) Are the rules internally consistent?
- Are there prompt-injection or identity-confusion paths still open? The
  corpus contains co-authors' names and contact details — that is the known
  hazard.
- Is `max_tokens=1024` ever truncating answers?

### E. Latency and cost
- Where does wall-clock time actually go? Instrument if needed.
- What is the p50/p95 uncached? What fraction of realistic traffic would the
  FAQ cache absorb?
- Is anything eagerly loaded that could be lazy, or recomputed that could be
  cached?
- Estimate per-question Anthropic token cost and where it is avoidable.

### F. Evaluation methodology
- Is a 20-item keyword/substring golden set adequate? Where does it mislead?
  (Known: it has failed answers that were correct but phrased differently, and
  passed an answer that was correct for the wrong reason.)
- Propose a better harness: more items, adversarial and unanswerable cases,
  LLM-judged faithfulness/answer-relevancy, retrieval-only metrics separated
  from generation metrics.
- Is the CI gate in `gate.py` measuring something that would actually catch a
  regression?

### G. Correctness, robustness, dead code
- Any remaining dead modules, unused config, or duplicated logic?
- Any failure path that returns a wrong answer instead of failing loudly?
- Are the tests meaningful, or do they assert trivia? Do the fakes faithfully
  mimic the real API shapes? (A previous test fixture stubbed `.text` to a
  placeholder and silently failed to exercise the error path.)
- Any correctness bug in ingest → index → retrieve that only appears when the
  corpus changes size or content?

---

## RULES YOU MUST FOLLOW

1. **Do not tune to the 20 eval questions.** Any change whose justification is
   "this makes q10/q20 pass" is rejected. If a change happens to fix them
   *because it fixes a general defect*, explain the general defect. State
   explicitly, for each proposal, why it would help unseen questions.
2. **One variable at a time.** If you propose several changes, specify the
   order to apply and measure them in, and say what metric movement would
   confirm or refute each. Do not bundle.
3. **Verify before asserting.** Read the code. Inspect the data. Run things.
   If a claim rests on something you could not execute here, label it
   `UNVERIFIED` and say what would verify it.
4. **Check what your fix breaks, not just what it fixes.** Every proposal must
   name a plausible regression and how you would detect it. (Item 5 in the
   fixed-bugs list is exactly this failure.)
5. **Respect the constraints.** Free tier, no GPU, no torch, tiny corpus,
   single provider, grounded answers.
6. **Distinguish "wrong" from "differently phrased".** Before calling an answer
   inaccurate, read the actual generated answer, not just the metric.
7. **No silent scope creep.** Do not refactor for taste.

---

## OUTPUT FORMAT

Produce a single markdown report with these sections **in this order**.

### 1. Executive summary
Five sentences max. The single most impactful finding, the current honest
health of the system, and your top recommendation.

### 2. Verified findings
Every issue you confirmed. Ordered by impact. Use exactly this structure:

```
#### F-01 — <short title>
Severity:      critical | high | medium | low
Category:      retrieval | chunking | generation | latency | eval | correctness
Evidence:      <file:line, command output, measured number — be concrete>
Root cause:    <the mechanism, not the symptom>
Why it matters
for unseen
questions:     <required — generality argument>
Proposed fix:  <specific change>
Risk / what
it could break:<required>
How to verify: <the measurement that confirms it worked>
Effort:        S | M | L
```

### 3. Unverified hypotheses
Things you suspect but could not confirm here, and exactly what would confirm
each. Keep these clearly separated from section 2.

### 4. Prioritised roadmap
A table: Order | Change | Expected effect | Metric that proves it | Effort |
Risk. Sequenced so each step is independently measurable.

### 5. Efficiency options (accuracy-neutral)
Latency and cost reductions that should not change answers, each with an
estimated saving and the evidence behind the estimate.

### 6. Things I deliberately did NOT recommend
Ideas you considered and rejected, with the reason (usually: overfitting,
constraint violation, or cost > benefit at 92 chunks). This section is
mandatory — it tells the reader what tradeoffs were already weighed.

### 7. Open questions for the owner
Decisions that need a human: product intent, acceptable latency, willingness
to spend, corpus additions.

---

## EXAMPLES — the standard expected

**GOOD finding** (specific, evidenced, root-caused, generalises, self-critical):

> #### F-03 — Icon labels deleted from résumé contact block cost BM25 its only
> match term
> Severity: high · Category: retrieval
> Evidence: `storage/chunks.jsonl` chunk 63 tokenizes to
> `['rudra','joshi',...,'61','466988797']`; the token `phone` is absent.
> Offline BM25 over the current corpus ranks this chunk #1 for "what is my
> phone number?" when the label is preserved and unranked when it is removed.
> Root cause: `_clean_pdf_text` deleted icon labels wholesale; the label was
> the only lexical bridge between a natural-language contact question and a
> chunk that otherwise contains only digits.
> Why it matters for unseen questions: every contact-shaped question ("how do
> I reach him", "mobile?") depends on that bridge; the résumé contains no
> other occurrence of these field words.
> Proposed fix: rewrite labels to `Phone: …` / `Email: …` rather than delete.
> Risk: over-eager rewriting could corrupt real URLs (`https://` lost a slash
> in an earlier attempt); guard with a lookbehind and assert clean text passes
> through unchanged.
> How to verify: re-ingest; assert `phone`/`email` tokens present in the
> contact chunk; re-run eval and confirm hit-rate does not drop.
> Effort: S

**BAD finding** (vague, unevidenced, ungeneralisable, unfalsifiable):

> Retrieval could be better. Consider using a bigger embedding model, adding
> more chunks, and increasing top_k. Also the prompt could be improved. This
> should fix the two failing questions.

Why it is bad: no evidence, no root cause, no risk analysis, no measurement,
bundles four changes, and is justified by the two known-failing eval items.

---

## FIRST STEPS

1. Read `src/rag/rag.py`, `retrieve_custom.py`, `prompt.py`,
   `llm_anthropic.py`, `chunking.py`, `doc_profiles.py`, `faq_cache.py`.
2. Inspect real data: `storage/chunks.jsonl` (sizes, boundaries, noise,
   per-source distribution) and `data/curated/*.md`.
3. Run the test suite: `python -m pytest tests/ -q`.
4. Reproduce a baseline: `python -m src.eval.run_eval` (needs
   `ANTHROPIC_API_KEY`; costs a small amount of credit).
5. For retrieval-only experiments, do **not** call the LLM — import
   `retrieve()` directly and measure hit/recall offline. It is free and fast.

Begin with a short plan of what you intend to inspect and measure, then
execute it.
