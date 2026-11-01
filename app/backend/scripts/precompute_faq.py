#!/usr/bin/env python
"""
Precompute answers for common questions so they serve instantly at runtime.

Run AFTER ingestion, whenever the source documents change:

    python -m scripts.precompute_faq
    python -m scripts.precompute_faq --only "Tell me about PersonaQuery."

Because this runs offline, a rate limit is a non-event: it waits and
retries instead of degrading a live visitor's answer. That's the whole
trade — move the quota risk from request time to build time.

Questions come from two places:
  - src/eval/qa.jsonl (the golden set — by definition the things worth
    getting right, and already curated)
  - FAQ_EXTRA below (phrasings a recruiter would actually type, which the
    eval set doesn't cover because it was written for measurement)

FAQ_OVERRIDES replaces the generated answer for questions where the model
reliably picks a partial answer; those are never sent to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.core.config import RAG_STORAGE_DIR as STORAGE_DIR
from src.rag.embedder import DEFAULT_EMBED_MODEL, get_embedder
from src.rag.faq_cache import CHUNKS_PATH, index_sha
from src.rag.rag import run_rag

OUT_PATH = Path(STORAGE_DIR) / "faq_cache.json"
QA_PATH = Path(__file__).resolve().parents[1] / "src" / "eval" / "qa.jsonl"

# Pacing between calls. Generous on purpose: this is a build step, nobody is
# waiting on it, and being gentle here avoids tripping provider rate limits
# mid-run.
PACE_S = float(os.getenv("FAQ_PACE_S", "6.0"))
MAX_ATTEMPTS = 3

# Real-visitor phrasings. The eval set is written for measurement ("What is
# my email address?" — first person, because it tests the identity rules);
# actual visitors ask in second/third person.
FAQ_EXTRA = [
    "How can I contact Rudra?",
    "What is Rudra's email?",
    "Where is Rudra based?",
    "Is Rudra authorised to work in Australia?",
    "What is Rudra's educational background?",
    "What did Rudra study?",
    "What are Rudra's technical skills?",
    "What programming languages does Rudra know?",
    "Does Rudra have experience with RAG or LLMs?",
    "What projects has Rudra built?",
    "Tell me about PersonaQuery.",
    "Tell me about HealthEcho.",
    "Tell me about JobPilot.",
    "What is the Cognitive-Chair project?",
    "Does Rudra have any patents?",
    "Has Rudra published any research?",
    "What awards has Rudra won?",
    "Which LLM does this chatbot use?",
    "What work experience does Rudra have?",
    "What internships has Rudra done?",
    "What kind of roles is Rudra looking for?",
    "Why should we hire Rudra?",
    "Summarise Rudra's professional profile.",
    "What is Rudra's experience with Python?",
    "Does Rudra have cloud or DevOps experience?",
    # Suggestion chips in the portfolio's Ask AI drawer (AskAI.js) and the
    # project cards' prompts (Projects.js), so a chip click is instant.
    "Who is Rudra, in one paragraph?",
    "What roles is he the best fit for?",
    "What is he working on right now?",
    "Explain the patent in plain English.",
    "How deep is his ML experience, really?",
    "Which stack is he strongest in?",
    "Has he shipped anything to production?",
    "Which project was the hardest, and why?",
    "Walk me through the research paper.",
    "Summarise his experience for a recruiter.",
    "What are his standout achievements?",
    "Explain the Cognitive Chair project and how the EEG/EOG control works.",
    "How does PersonaQuery's RAG pipeline work under the hood?",
    "Summarise the Cognitive Chair research paper — methodology and results.",
    "Tell me about HealthEcho — what was the hardest part to build?",
    "What did the PPO vs SAC vs TD3 comparison actually find?",
]

# Hand-written answers. Sources are (file_name, section prefix) and are
# resolved against the CURRENT index at build time, so snippets never go
# stale; a missing section fails the build rather than citing nothing.
# [[cite:N]] refers to the N-th source in the list.

# Shared by both patent chips. Generated answers invented how the system
# works and tied it to the Cognitive-Chair paper; it is a separate work.
_PATENT = {
    "answer": (
        "The patent application is titled \"Intelligent Face Recognition "
        "System using Deep Recurrent Neural Network\" and was filed in India as "
        "application 202211058943 A on 15 October 2022 [[cite:1]]. It was "
        "published in Patent Office Journal No. 42/2022, dated 21 October 2022 "
        "[[cite:1]]. His documents list it as a separate work from the "
        "Cognitive-Chair paper [[cite:2]].\n\n"
        "Not stated in the documents: what the application claims, or how the "
        "system works beyond its title."
    ),
    "sources": [
        ("awards-and-certifications.md", "Intelligent Face Recognition"),
        ("linkedin-profile.md", "Publications and Patents", "separate works"),
    ],
}

# Shared by both education chips. Generated answers missed the Master's
# (retrieval skipped education.md for these wordings). Years only: the
# documents disagree on the months.
_EDUCATION = {
    "answer": (
        "Rudra holds a Master of Computer Science from the University of "
        "Wollongong, majoring in Machine Learning and Big Data, with a UOW "
        "Postgraduate Academic Excellence Scholarship (30% tuition fee waiver) "
        "[[cite:1]]. Before that he completed a B.Tech in Computer Science & "
        "Engineering (Data Science Specialisation) at GSFC University, "
        "2019–2023 [[cite:2]], and received the Achiever's Award for academic "
        "year 2021–22 [[cite:3]]."
    ),
    "sources": [
        ("education.md", "Master of Computer Science"),
        ("Rudra_CV.pdf", "EDUCATION"),
        ("awards-and-certifications.md", "Achiever"),
    ],
}

FAQ_OVERRIDES = {
    "What is Rudra's educational background?": _EDUCATION,
    "What did Rudra study?": _EDUCATION,
    # Generated version compared him to "a typical first-year graduate",
    # which no document supports. Facts only, no comparison.
    "How deep is his ML experience, really?": {
        "answer": (
            "The documents describe his ML experience through his degree, his "
            "projects and one research paper. He is seeking his first commercial "
            "AI/ML or software engineering role [[cite:1]].\n\n"
            "- **Degree:** Master of Computer Science at the University of "
            "Wollongong, majoring in Machine Learning and Big Data [[cite:2]].\n"
            "- **Retrieval and evaluation (PersonaQuery):** hybrid retrieval with "
            "`BAAI/bge-small-en-v1.5` dense embeddings via FastEmbed, BM25 keyword "
            "search, Reciprocal Rank Fusion and ColBERT late-interaction "
            "reranking, over a NumPy matrix with no external vector database "
            "[[cite:3]]. An 18-item golden set measures retrieval hit-rate and "
            "recall@k, answer keyword match and citation compliance. The CI gate "
            "fails the build if retrieval hit-rate@8 drops below 0.90 [[cite:4]].\n"
            "- **Deep reinforcement learning (benchmarking study):** a comparative "
            "analysis of DDPG, TD3, SAC and PPO in continuous action spaces. It "
            "focuses on reward convergence, training stability and sample "
            "efficiency across random seeds, using OpenAI Gym and MuJoCo "
            "environments [[cite:5]].\n"
            "- **LLM agents (JobPilot):** a 6-stage job-application pipeline with "
            "LLM normalisation and 0–100 match scoring (Claude Haiku, structured "
            "output), and actor–critic cover-letter generation (Claude Opus) with "
            "a 14-point QA gate [[cite:6]].\n"
            "- **Research:** co-author of the Cognitive-Chair paper, on an "
            "AI-based brain-sensing wheelchair controlled through EEG and EOG "
            "signals. It won the Best Paper Award at IEEE AIST 2022 [[cite:7]].\n\n"
            "The documents do not describe commercial or team-based ML work "
            "[[cite:1]], and they do not state the results of the reinforcement "
            "learning study."
        ),
        "sources": [
            ("Rudra_CV.pdf", "SUMMARY"),
            ("education.md", "Master of Computer Science"),
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("projects.md", "PersonaQuery", "drops below 0.90"),
            ("projects.md", "Deep Reinforcement Learning"),
            ("Rudra_CV.pdf", "KEYPROJECTS", "6-stage"),
            ("projects.md", "Cognitive-Chair", "EEG and EOG"),
        ],
    },
    # The model kept answering with the patent plus a quiz participation and
    # dropping the GSFC Achiever's Award.
    # Re-check this text whenever data/curated/awards-and-certifications.md changes.
    "What awards or patents does Rudra have?": {
        "answer": (
            "Rudra has two awards, one scholarship and one patent application.\n\n"
            "**Awards:**\n\n"
            "- **IEEE Best Paper Award, AIST 2022** - Best Paper at the 4th "
            "International Conference on Artificial Intelligence and Speech "
            "Technology, for the co-authored paper \"Cognitive-Chair: AI based "
            "advanced Brain Sensing Wheelchair for Paraplegic/Quadriplegic "
            "people\", published on IEEE Xplore [[cite:1]].\n"
            "- **Achiever's Award, GSFC University** - awarded by the School of "
            "Technology for the B.Tech. Computer Science and Engineering "
            "programme, academic year 2021-22, conferred 26 January 2023 "
            "[[cite:2]].\n\n"
            "**Scholarship:**\n\n"
            "- **UOW Postgraduate Academic Excellence Scholarship (2023)** - a "
            "30% tuition fee waiver for the Master of Computer Science at the "
            "University of Wollongong [[cite:3]].\n\n"
            "**Patent:**\n\n"
            "- **Intelligent Face Recognition System using Deep Recurrent Neural "
            "Network** - Indian patent application 202211058943 A, filed "
            "15 October 2022 [[cite:4]]."
        ),
        "sources": [
            ("awards-and-certifications.md", "IEEE Best Paper"),
            ("awards-and-certifications.md", "Achiever"),
            ("awards-and-certifications.md", "UOW Postgraduate"),
            ("awards-and-certifications.md", "Intelligent Face Recognition"),
        ],
    },
    # The generated answer pitched him for "Senior" roles; the resume says
    # he is seeking a first commercial role. First-role framing, facts only.
    # Re-check this text whenever the resume or curated docs change.
    "Why should we hire Rudra?": {
        "answer": (
            "Rudra holds a Master of Computer Science, has full Australian work "
            "rights and is seeking his first commercial AI/ML or software "
            "engineering role [[cite:1]].\n\n"
            "- **Education:** Master of Computer Science at the University of "
            "Wollongong, majoring in Machine Learning and Big Data, with a UOW "
            "Postgraduate Academic Excellence Scholarship (30% tuition fee "
            "waiver) [[cite:2]].\n"
            "- **Built and deployed a RAG service end-to-end:** PersonaQuery "
            "answers questions from his own documents with inline citations and "
            "prompt-injection guardrails, using hybrid BM25 and dense retrieval "
            "fused with Reciprocal Rank Fusion and ColBERT reranking. It has a "
            "FastAPI backend on Google Cloud Run [[cite:3]].\n"
            "- **Measures what he builds:** PersonaQuery has a golden-set "
            "evaluation harness and a CI pipeline that runs ruff, the full "
            "pytest suite and a retrieval-quality gate on every push [[cite:4]]. "
            "The gate fails the build if retrieval hit-rate@8 drops below 0.90 "
            "on the 18-item golden set [[cite:5]].\n"
            "- **Multi-agent automation:** JobPilot is a 6-stage job-application "
            "pipeline covering multi-source scraping, LLM match scoring, "
            "tailored resume and cover-letter generation, and Playwright form "
            "filling [[cite:6]].\n"
            "- **Research:** co-author of the Cognitive-Chair paper, which won "
            "the Best Paper Award at IEEE AIST 2022 and is published on IEEE "
            "Xplore [[cite:7]].\n"
            "- **Patent:** Indian patent application 202211058943 A for an "
            "intelligent face recognition system using a deep recurrent neural "
            "network, filed 15 October 2022 [[cite:8]].\n\n"
            "He is based in Sydney and open to relocation [[cite:9]]."
        ),
        "sources": [
            ("Rudra_CV.pdf", "SUMMARY"),
            ("education.md", "Master of Computer Science"),
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("projects.md", "PersonaQuery", "full pytest suite"),
            ("projects.md", "PersonaQuery", "drops below 0.90"),
            ("Rudra_CV.pdf", "KEYPROJECTS", "6-stage"),
            ("awards-and-certifications.md", "IEEE Best Paper"),
            ("awards-and-certifications.md", "Intelligent Face Recognition"),
            ("Rudra_CV.pdf", "RUDRA JOSHI", "open to relocation"),
        ],
    },
    # Generated version said "co-filed a patent" (it is an application) and
    # skipped the Master's.
    "Who is Rudra, in one paragraph?": {
        "answer": (
            "Rudra Joshi holds a Master of Computer Science from the University "
            "of Wollongong, majoring in Machine Learning and Big Data, where he "
            "received a UOW Postgraduate Academic Excellence Scholarship (30% "
            "tuition fee waiver) [[cite:1]]. He is based in Sydney [[cite:2]]. "
            "He has full Australian work rights and is seeking his first "
            "commercial AI/ML or software engineering role [[cite:3]]. He built "
            "and deployed PersonaQuery end-to-end: a retrieval-augmented "
            "assistant that answers questions from his own documents with "
            "inline citations, with a FastAPI backend on Google Cloud Run "
            "[[cite:4]]. He co-authored the Cognitive-Chair paper, which won the "
            "Best Paper Award at IEEE AIST 2022 [[cite:5]]. Separately, his "
            "documents list Indian patent application 202211058943 A, "
            "\"Intelligent Face Recognition System using Deep Recurrent Neural "
            "Network\" [[cite:6]]."
        ),
        "sources": [
            ("education.md", "Master of Computer Science"),
            ("Rudra_CV.pdf", "RUDRA JOSHI", "Sydney"),
            ("Rudra_CV.pdf", "SUMMARY"),
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("awards-and-certifications.md", "IEEE Best Paper"),
            ("awards-and-certifications.md", "Intelligent Face Recognition"),
        ],
    },
    "Explain the patent in plain English.": _PATENT,
    "Explain the patent application in plain English — what does it claim?": _PATENT,
    # Generated version quoted the LinkedIn headline's LangChain claim, which
    # the repo doesn't back; this cites the About section instead.
    "What roles is he the best fit for?": {
        "answer": (
            "Rudra is seeking his first commercial role, in AI/ML or software "
            "engineering [[cite:1]]. His LinkedIn profile says he is open to AI "
            "Engineer, ML Engineer, GenAI Engineer, LLM Engineer and Applied AI "
            "roles in Sydney, with full Australian work rights [[cite:2]].\n\n"
            "The evidence for those roles:\n\n"
            "- **Retrieval and LLM systems:** he built and deployed PersonaQuery, "
            "a RAG assistant with hybrid BM25 and dense retrieval, Reciprocal "
            "Rank Fusion, ColBERT reranking and inline citations, with a FastAPI "
            "backend on Google Cloud Run [[cite:3]].\n"
            "- **Evaluation and testing:** PersonaQuery has a golden-set "
            "evaluation harness and a CI pipeline that runs ruff, the full "
            "pytest suite and a retrieval-quality gate on every push [[cite:4]].\n"
            "- **Multi-agent automation:** JobPilot is a 6-stage job-application "
            "pipeline covering multi-source scraping, LLM match scoring, "
            "tailored resume and cover-letter generation, and Playwright form "
            "filling [[cite:5]].\n"
            "- **ML foundations:** a Master of Computer Science at the University "
            "of Wollongong, majoring in Machine Learning and Big Data [[cite:6]].\n\n"
            "He is based in Sydney and open to relocation [[cite:7]]."
        ),
        "sources": [
            ("Rudra_CV.pdf", "SUMMARY"),
            ("linkedin-profile.md", "About", "Open to"),
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("projects.md", "PersonaQuery", "full pytest suite"),
            ("Rudra_CV.pdf", "KEYPROJECTS", "6-stage"),
            ("education.md", "Master of Computer Science"),
            ("Rudra_CV.pdf", "RUDRA JOSHI", "open to relocation"),
        ],
    },
    # Generated version listed LangChain/LlamaIndex/ChromaDB from the stale
    # CV skills line. The documents don't rank skills; list what projects use.
    "Which stack is he strongest in?": {
        "answer": (
            "The documents don't rank his skills, so this is what his own "
            "projects use.\n\n"
            "- **Python backends and retrieval:** PersonaQuery has a FastAPI "
            "backend on Google Cloud Run. Its retrieval is hand-built: "
            "`BAAI/bge-small-en-v1.5` embeddings via FastEmbed, BM25 keyword "
            "search, Reciprocal Rank Fusion and ColBERT reranking, over a NumPy "
            "matrix with no external vector database. Generation runs on "
            "Qwen3.8-27B on Groq [[cite:1]].\n"
            "- **LLM agents:** JobPilot is built with Python, the Anthropic SDK, "
            "Playwright, Pydantic and Tenacity [[cite:2]].\n"
            "- **JavaScript frontends:** his portfolio is a React 18 site with "
            "React-Bootstrap, GSAP and an Express + Nodemailer contact server "
            "[[cite:3]]. HealthEcho has a Next.js 15 frontend with React, "
            "TypeScript and Tailwind CSS, and an Express.js backend that calls "
            "Together AI's Mixtral-8x7B [[cite:4]].\n\n"
            "His LinkedIn profile says \"I write clean Python\" [[cite:5]]."
        ),
        "sources": [
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("projects.md", "JobPilot", "Tenacity"),
            ("projects.md", "Personal portfolio", "React 18"),
            ("projects.md", "HealthEcho"),
            ("linkedin-profile.md", "About", "clean Python"),
        ],
    },
    # Generated version said "patent holder" (from the stale CV summary).
    "Summarise his experience for a recruiter.": {
        "answer": (
            "**Summary:** Master of Computer Science with full Australian work "
            "rights, seeking a first commercial AI/ML or software engineering "
            "role [[cite:1]]. Based in Sydney and open to relocation [[cite:2]].\n\n"
            "**Education:** Master of Computer Science at the University of "
            "Wollongong, majoring in Machine Learning and Big Data, with a UOW "
            "Postgraduate Academic Excellence Scholarship [[cite:3]]. B.Tech in "
            "Computer Science & Engineering (Data Science Specialisation) at "
            "GSFC University, 2019-2023 [[cite:4]].\n\n"
            "**Experience:**\n\n"
            "- Android Application Development Intern, TSPL, Vadodara (Jan 2023 "
            "- Jul 2023): built an Android app end-to-end with backend REST APIs "
            "and analytics charts, in Java/Spring Boot and MySQL [[cite:5]].\n"
            "- Engineering Intern (Full Stack + Multi-Domain), Gurukula, Tntra "
            "Engineering, Vadodara (Jun 2022 - Dec 2022): built and shipped web "
            "app modules, and completed multi-domain training in web dev, cloud "
            "and security [[cite:5]].\n"
            "- Independent AI Engineer since Jul 2023, building PersonaQuery and "
            "HealthEcho end-to-end [[cite:6]].\n\n"
            "**Projects:**\n\n"
            "- **PersonaQuery:** a RAG assistant with hybrid BM25 and dense "
            "retrieval, Reciprocal Rank Fusion, ColBERT reranking and inline "
            "citations, with a FastAPI backend on Google Cloud Run [[cite:7]].\n"
            "- **JobPilot:** a 6-stage multi-agent job-application pipeline "
            "[[cite:8]].\n"
            "- **HealthEcho:** a symptom-analysis chatbot, live at "
            "https://healthecho.vercel.app/ [[cite:9]].\n\n"
            "**Research and IP:** co-author of the Cognitive-Chair paper (Best "
            "Paper Award, IEEE AIST 2022) [[cite:10]]. Separately, Indian patent "
            "application 202211058943 A [[cite:11]]."
        ),
        "sources": [
            ("Rudra_CV.pdf", "SUMMARY"),
            ("Rudra_CV.pdf", "RUDRA JOSHI", "open to relocation"),
            ("education.md", "Master of Computer Science"),
            ("Rudra_CV.pdf", "EDUCATION"),
            ("Rudra_CV.pdf", "EXPERIENCE"),
            ("linkedin-profile.md", "Independent AI Engineer"),
            ("projects.md", "PersonaQuery", "Repository layout"),
            ("Rudra_CV.pdf", "KEYPROJECTS", "6-stage"),
            ("projects.md", "HealthEcho"),
            ("awards-and-certifications.md", "IEEE Best Paper"),
            ("awards-and-certifications.md", "Intelligent Face Recognition"),
        ],
    },
    # Generated version implied a granted patent.
    "What are his standout achievements?": {
        "answer": (
            "- **IEEE Best Paper Award, AIST 2022:** for the co-authored paper "
            "\"Cognitive-Chair: AI based advanced Brain Sensing Wheelchair for "
            "Paraplegic/Quadriplegic people\", published on IEEE Xplore "
            "[[cite:1]].\n"
            "- **Indian patent application 202211058943 A:** \"Intelligent Face "
            "Recognition System using Deep Recurrent Neural Network\", filed "
            "15 October 2022 [[cite:2]].\n"
            "- **UOW Postgraduate Academic Excellence Scholarship (2023):** a 30% "
            "tuition fee waiver for his Master of Computer Science [[cite:3]].\n"
            "- **Achiever's Award, GSFC University:** for the B.Tech Computer "
            "Science and Engineering programme, academic year 2021-22 [[cite:4]].\n"
            "- **PersonaQuery:** built end-to-end, with a CI gate that fails the "
            "build if retrieval hit-rate@8 drops below 0.90 on an 18-item golden "
            "set [[cite:5]].\n"
            "- **HealthEcho:** live at https://healthecho.vercel.app/ [[cite:6]]."
        ),
        "sources": [
            ("awards-and-certifications.md", "IEEE Best Paper"),
            ("awards-and-certifications.md", "Intelligent Face Recognition"),
            ("awards-and-certifications.md", "UOW Postgraduate"),
            ("awards-and-certifications.md", "Achiever"),
            ("projects.md", "PersonaQuery", "drops below 0.90"),
            ("projects.md", "HealthEcho"),
        ],
    },
}


def _load_eval_questions() -> list[str]:
    if not QA_PATH.exists():
        return []
    out = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(json.loads(s)["question"])
            except Exception:
                continue
    return out


def _resolve_sources(refs: list[tuple[str, ...]]) -> list[dict]:
    # (file_name, section prefix[, text the chunk must contain]): long
    # sections split into several chunks, so a fact can pin the right one.
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    out = []
    for i, (file_name, prefix, *needles) in enumerate(refs, 1):
        c = next((c for c in chunks
                  if c["metadata"].get("file_name") == file_name
                  and (c["metadata"].get("section") or "").startswith(prefix)
                  and all(n in c["text"] for n in needles)), None)
        if c is None:
            raise SystemExit(f"FAQ_OVERRIDES: no chunk {file_name} / {prefix!r}; re-check after ingest")
        out.append({"id": i, "file_name": file_name, "page_label": "n/a",
                    "section": c["metadata"]["section"], "relevance": 1.0,
                    "channel": "override", "snippet": c["text"][:300]})
    return out


def _embed(embedder, q: str) -> list[float]:
    vec = np.array(embedder.embed_query(q), dtype="float32")
    return (vec / (np.linalg.norm(vec) + 1e-12)).tolist()


def _build_entry(q: str, embedder, label: str) -> dict | None:
    """One FAQ entry: the override if there is one, else a generated answer."""
    if q in FAQ_OVERRIDES:
        o = FAQ_OVERRIDES[q]
        print(f"  {label} override: {q}")
        return {"question": q, "answer": o["answer"],
                "sources": _resolve_sources(o["sources"]), "embedding": _embed(embedder, q)}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # use_faq=False: never read the cache we're in the middle of
        # rebuilding, or we'd just re-cache yesterday's answers.
        res = run_rag(q, mode="chat", use_faq=False)
        text = res.get("answer", "")
        # run_rag never raises; it degrades. Detect degraded output rather
        # than caching an apology or an extractive stopgap as if it were
        # a real answer.
        degraded = (
            "temporarily unavailable" in text.lower()
            or "having trouble generating" in text.lower()
            or not text.strip()
        )
        if not degraded:
            print(f"  {label} ok: {q}")
            return {"question": q, "answer": res["answer"],
                    "sources": res.get("sources", []), "embedding": _embed(embedder, q)}
        wait = 20 * attempt
        print(f"  {label} degraded (attempt {attempt}); waiting {wait}s")
        time.sleep(wait)
    print(f"  {label} FAILED: {q}")
    return None


def _write(entries: list[dict]) -> None:
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embed_model": DEFAULT_EMBED_MODEL,
            "index_sha": index_sha(),
            "entries": entries,
        }, f, ensure_ascii=False)


def _only(questions: list[str]) -> int:
    """Regenerate just these entries and merge them into the existing cache."""
    if not OUT_PATH.exists():
        print(f"{OUT_PATH} not found; run a full build first.")
        return 1
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    if data.get("index_sha") != index_sha():
        # The merged file is stamped with the CURRENT index, so the entries
        # left alone are trusted against it too. Fine when the re-ingest
        # didn't touch what they say; otherwise --only those as well.
        print("WARNING: cache was built for a different index; untouched "
              "entries are kept as-is and re-stamped.")
    embedder = get_embedder()
    entries, merged = data.get("entries", []), 0
    for i, q in enumerate(questions, 1):
        if i > 1 and q not in FAQ_OVERRIDES:
            time.sleep(PACE_S)
        entry = _build_entry(q, embedder, f"[{i}/{len(questions)}]")
        if entry is None:
            break  # still write what succeeded: each entry cost real quota
        keep = [e for e in entries if e["question"].strip().lower() != q.lower()]
        entries, merged = keep + [entry], merged + 1
    _write(entries)
    print(f"\nMerged {merged}/{len(questions)} -> {OUT_PATH} ({len(entries)} total)")
    return 0 if merged == len(questions) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", metavar="QUESTION",
                    help="regenerate just this question (repeatable) and merge it in")
    args = ap.parse_args(argv)
    if args.only:
        return _only([q.strip() for q in args.only if q.strip()])

    questions = _load_eval_questions() + FAQ_EXTRA + list(FAQ_OVERRIDES)
    # De-dupe case-insensitively, preserving order.
    seen, ordered = set(), []
    for q in questions:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            ordered.append(q.strip())

    print(f"Precomputing {len(ordered)} FAQ answers (pace={PACE_S}s)\n")

    embedder = get_embedder()
    entries, failed = [], []

    for i, q in enumerate(ordered, 1):
        if i > 1 and q not in FAQ_OVERRIDES:
            time.sleep(PACE_S)
        entry = _build_entry(q, embedder, f"[{i}/{len(ordered)}]")
        if entry is None:
            failed.append(q)
        else:
            entries.append(entry)

    _write(entries)

    print(f"\nWrote {len(entries)} entries -> {OUT_PATH}")
    if failed:
        print(f"{len(failed)} question(s) could not be generated "
              f"(providers exhausted). Re-run later to fill them in:")
        for q in failed:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
