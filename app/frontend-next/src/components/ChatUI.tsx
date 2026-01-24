"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";

type Source = {
  file_name?: string;
  page_label?: string;
  score?: number;
  snippet?: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const DEFAULT_PROMPTS = [
  "Top 3 GenAI projects",
  "Best-fit roles",
  "Key strengths (ATS)",
  "Publications & patent",
];

function buildApiUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_API_URL?.trim() || "http://127.0.0.1:8000";
  return base.replace(/\/+$/, "") + path;
}

export default function ChatUI() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chatRef = useRef<HTMLDivElement | null>(null);

  const isFirstScreen = messages.length === 0;

  useEffect(() => {
    // Auto-scroll when chat appears / updates
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function send(questionOverride?: string) {
    const q = (questionOverride ?? input).trim();
    if (!q || loading) return;

    setError(null);
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");

    try {
      const res = await fetch(buildApiUrl("/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // CORS preflight is handled by FastAPI CORSMiddleware
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const t = await res.text().catch(() => "");
        throw new Error(`API error (${res.status}): ${t || res.statusText}`);
      }

      const data = await res.json();

      // Adjust these keys if your backend uses different names
      const answer: string =
        typeof data?.answer === "string"
          ? data.answer
          : typeof data?.response === "string"
          ? data.response
          : typeof data?.text === "string"
          ? data.text
          : "";

      const sources: Source[] | undefined = Array.isArray(data?.sources) ? data.sources : undefined;

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer || "I didn’t get an answer back from the API. Check backend logs for /chat.",
          sources,
        },
      ]);
    } catch (e: any) {
      setError(e?.message || "Failed to fetch");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "⚠️ API error. Please check your backend is running and CORS is enabled. Also confirm NEXT_PUBLIC_API_URL.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden text-white">
      {/* Background */}
      <div
        className="absolute inset-0 -z-10"
        style={{
          backgroundImage: `linear-gradient(180deg, rgba(0,0,0,.62), rgba(0,0,0,.70)), url('/bg.png')`,
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />
      {/* Soft vignette */}
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_50%_30%,rgba(64,184,255,0.18),transparent_55%),radial-gradient(circle_at_30%_90%,rgba(120,60,255,0.16),transparent_55%)]" />

      {/* Header */}
      <header className="mx-auto flex max-w-6xl flex-col items-center px-4 pt-10">
        <div className="flex items-center justify-center gap-3">
          <div className="relative h-10 w-10 overflow-hidden rounded-xl">
            <Image src="/logo.png" alt="PersonaQuery" fill className="object-cover" priority />
          </div>

          <h1 className="text-4xl font-semibold tracking-wide">
            <span className="text-white">Persona</span>
            <span className="text-sky-300">Query</span>
          </h1>
        </div>

        <p className="mt-2 text-center text-sm text-white/70">
          Ask questions about Rudra’s resume, patent, and research paper (RAG + Groq).
        </p>
      </header>

      {/* Main */}
      <main className="mx-auto flex max-w-6xl flex-col px-4 pb-28 pt-8">
        {/* Landing content (ONLY before first message) */}
        {isFirstScreen && (
          <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-[420px_1fr]">
            {/* Welcome card */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur-xl">
              <div className="text-sm font-semibold">Welcome, Recruiter / Guest 👋</div>
              <div className="mt-3 text-sm leading-relaxed text-white/80">
                I’m PersonaQuery — Rudra’s digital twin. Ask about skills, projects, achievements, publications, and
                experience. I only answer using evidence from the provided documents.
              </div>
            </div>

            {/* Prompt chips */}
            <div className="rounded-2xl border border-white/10 bg-white/0 p-2">
              <div className="mb-3 text-center text-xs uppercase tracking-wider text-white/50">
                Quick prompts (click one)
              </div>

              <div className="flex flex-wrap items-center justify-center gap-3">
                {DEFAULT_PROMPTS.map((p) => (
                  <button
                    key={p}
                    onClick={() => void send(p)}
                    className="rounded-full border border-white/15 bg-white/5 px-5 py-2 text-sm text-white/85 shadow-[0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur-md transition hover:bg-white/10 hover:text-white"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Chat panel (ONLY after first message) */}
        {!isFirstScreen && (
          <div
            ref={chatRef}
            className="mt-6 h-[62vh] w-full overflow-y-auto rounded-3xl border border-white/12 bg-white/5 p-6 shadow-[0_30px_90px_rgba(0,0,0,0.45)] backdrop-blur-2xl"
          >
            <div className="space-y-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex w-full ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={[
                      "max-w-[78%] rounded-2xl border border-white/12 px-4 py-3 backdrop-blur-xl",
                      "shadow-[0_18px_55px_rgba(0,0,0,0.35)]",
                      m.role === "user"
                        ? "bg-[linear-gradient(135deg,rgba(40,180,255,0.20),rgba(120,60,255,0.18))]"
                        : "bg-white/7",
                    ].join(" ")}
                  >
                    <div className="whitespace-pre-wrap text-sm leading-relaxed text-white/90">{m.content}</div>

                    {/* Sources (render properly; no object-as-child crash) */}
                    {m.role === "assistant" && Array.isArray(m.sources) && m.sources.length > 0 && (
                      <details className="mt-3 rounded-xl border border-white/10 bg-black/10 px-3 py-2">
                        <summary className="cursor-pointer text-xs text-white/70">Sources</summary>
                        <ul className="mt-2 space-y-2 text-xs text-white/70">
                          {m.sources.slice(0, 6).map((s, i) => (
                            <li key={i} className="rounded-lg border border-white/10 bg-white/5 p-2">
                              <div className="font-medium text-white/80">
                                {s.file_name || "Document"}{" "}
                                {s.page_label ? <span className="text-white/60">({s.page_label})</span> : null}
                                {typeof s.score === "number" ? (
                                  <span className="ml-2 text-white/50">score {s.score.toFixed(3)}</span>
                                ) : null}
                              </div>
                              {s.snippet ? (
                                <div className="mt-1 line-clamp-3 whitespace-pre-wrap text-white/65">{s.snippet}</div>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/70 backdrop-blur-xl">
                    Thinking…
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error line */}
        {error && <div className="mt-3 text-center text-xs text-red-200/90">{error}</div>}
      </main>

      {/* Input bar (always bottom, single glassy bar, send button inside) */}
      <div className="fixed bottom-6 left-0 right-0 z-20 px-4">
        <div className="mx-auto flex max-w-6xl items-center gap-3 rounded-full border border-white/14 bg-white/6 px-5 py-3 shadow-[0_25px_80px_rgba(0,0,0,0.55)] backdrop-blur-2xl">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your question..."
            className="w-full bg-transparent text-sm text-white/90 outline-none placeholder:text-white/45"
          />

          <button
            type="button"
            onClick={() => void send()}
            disabled={!canSend}
            className={[
              "flex h-11 w-14 items-center justify-center rounded-full",
              "border border-white/14 bg-white/8 backdrop-blur-xl",
              "shadow-[0_12px_35px_rgba(0,0,0,0.45)] transition",
              canSend ? "hover:bg-white/12" : "opacity-50 cursor-not-allowed",
            ].join(" ")}
            aria-label="Send"
          >
            <span className="text-lg">➤</span>
          </button>
        </div>
      </div>
    </div>
  );
}
