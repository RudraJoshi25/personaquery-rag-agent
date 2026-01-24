import os
import json
import base64
import requests
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------
def _b64_img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_api(question: str) -> dict:
    api_url = st.session_state.get("api_url") or os.getenv("PERSONAQUERY_API_URL", "http://127.0.0.1:8000")
    api_url = api_url.rstrip("/")
    r = requests.post(f"{api_url}/chat", json={"question": question}, timeout=120)
    r.raise_for_status()
    return r.json()

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role":"user|assistant","content":"..."}]
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("PERSONAQUERY_API_URL", "http://127.0.0.1:8000")
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def submit_message():
    q = (st.session_state.user_input or "").strip()
    if not q:
        return

    # add user message
    st.session_state.messages.append({"role": "user", "content": q})

    # call backend
    try:
        data = call_api(q)
        answer = data.get("answer", "").strip()
        if not answer:
            answer = "I couldn’t generate an answer from the current context. Try asking more specifically."
    except Exception as e:
        answer = f"⚠️ API error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # clear input
    st.session_state.user_input = ""

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PersonaQuery", page_icon="🧠", layout="wide")
init_state()

# -----------------------------
# Assets
# -----------------------------
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
BG_PATH = os.path.join(ASSET_DIR, "bg.png")
LOGO_PATH = os.path.join(ASSET_DIR, "logo.png")

bg_b64 = _b64_img(BG_PATH) if os.path.exists(BG_PATH) else ""
logo_b64 = _b64_img(LOGO_PATH) if os.path.exists(LOGO_PATH) else ""

# -----------------------------
# CSS (Glassy UI)
# -----------------------------
st.markdown(
    f"""
<style>
/* ===== Page background ===== */
.stApp {{
  background: url("data:image/png;base64,{bg_b64}") center/cover no-repeat fixed;
}}

/* Remove default padding feel */
.block-container {{
  padding-top: 16px !important;
  padding-bottom: 120px !important; /* space for fixed input bar */
  max-width: 1180px;
}}

/* Hide sidebar completely */
section[data-testid="stSidebar"] {{
  display: none !important;
}}

/* ===== Center header ===== */
.pq-header {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  margin-top: 10px;
  margin-bottom: 6px;
}}
.pq-logo {{
  width: 40px; height: 40px;
  border-radius: 10px;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.12), 0 10px 30px rgba(0,0,0,0.35);
}}
.pq-title {{
  font-size: 34px;
  font-weight: 800;
  letter-spacing: 0.3px;
  line-height: 1;
}}
.pq-title .persona {{ color: rgba(255,255,255,0.95); }}
.pq-title .query {{ color: rgba(79, 188, 255, 0.95); }}
.pq-subtitle {{
  text-align: center;
  color: rgba(220,235,255,0.75);
  font-size: 13px;
  margin-bottom: 16px;
}}

/* ===== Welcome + prompt cards ===== */
.glass-card {{
  background: linear-gradient(180deg, rgba(25,45,60,0.55), rgba(10,20,30,0.35));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 16px;
  backdrop-filter: blur(12px);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}}
.welcome-title {{
  font-weight: 700;
  color: rgba(255,255,255,0.92);
  margin-bottom: 8px;
}}
.welcome-text {{
  color: rgba(225,240,255,0.78);
  font-size: 13px;
  line-height: 1.45;
}}
.prompts-wrap {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  align-items: center;
}}
.pq-chip {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 10px 16px;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(30,70,90,0.55), rgba(10,25,35,0.35));
  border: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(10px);
  color: rgba(240,250,255,0.9);
  font-size: 13px;
  cursor: pointer;
  user-select: none;
  transition: transform 120ms ease, box-shadow 120ms ease;
}}
.pq-chip:hover {{
  transform: translateY(-1px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}}

/* ===== Chat panel (ONLY after first message) ===== */
.chat-panel {{
  margin-top: 14px;
  background: linear-gradient(180deg, rgba(25,45,60,0.55), rgba(10,20,30,0.35));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px;
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}}
.msg-row {{
  display: flex;
  margin: 10px 0;
}}
.msg-user {{
  justify-content: flex-end;
}}
.msg-assistant {{
  justify-content: flex-start;
}}
.bubble {{
  max-width: 74%;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 14px;
  line-height: 1.45;
}}
.bubble.user {{
  background: linear-gradient(180deg, rgba(38,110,140,0.55), rgba(15,40,55,0.35));
  color: rgba(245,252,255,0.96);
}}
.bubble.assistant {{
  background: linear-gradient(180deg, rgba(25,45,60,0.55), rgba(10,20,30,0.35));
  color: rgba(235,245,255,0.90);
}}

/* ===== Bottom input bar (glassy only, no black wrapper) ===== */
/* Make Streamlit form container transparent */
div[data-testid="stForm"] {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}}

/* Fixed bottom container */
.pq-inputbar {{
  position: fixed;
  left: 0;
  right: 0;
  bottom: 18px;
  display: flex;
  justify-content: center;
  z-index: 9999;
  pointer-events: none; /* only children clickable */
}}
.pq-input-inner {{
  width: min(1180px, calc(100vw - 40px));
  display: grid;
  grid-template-columns: 1fr 54px;
  gap: 12px;
  padding: 14px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(10,18,24,0.35), rgba(10,18,24,0.18));
  border: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(16px);
  box-shadow: 0 20px 80px rgba(0,0,0,0.45);
  pointer-events: auto; /* clickable */
}}

/* Style text input */
div[data-testid="stTextInput"] input {{
  height: 46px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: linear-gradient(180deg, rgba(30,70,90,0.40), rgba(10,25,35,0.22)) !important;
  color: rgba(245,252,255,0.94) !important;
  padding-left: 14px !important;
}}
div[data-testid="stTextInput"] input:focus {{
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(79,188,255,0.18) !important;
  border-color: rgba(79,188,255,0.32) !important;
}}

/* Style send button */
div[data-testid="stFormSubmitButton"] button {{
  height: 46px !important;
  width: 54px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(180deg, rgba(35,85,110,0.55), rgba(12,30,42,0.35)) !important;
  color: rgba(240,250,255,0.95) !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.28);
}}
div[data-testid="stFormSubmitButton"] button:hover {{
  transform: translateY(-1px);
}}
/* Remove Streamlit "label spacing" */
label {{
  display: none !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header (Centered)
# -----------------------------
st.markdown(
    f"""
<div class="pq-header">
  {"<img class='pq-logo' src='data:image/png;base64," + logo_b64 + "' />" if logo_b64 else ""}
  <div class="pq-title"><span class="persona">Persona</span><span class="query">Query</span></div>
</div>
<div class="pq-subtitle">Ask questions about Rudra’s resume, patent, and research paper (RAG + Groq).</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# HERO section (only before first question)
# -----------------------------
has_chat = len(st.session_state.messages) > 0

if not has_chat:
    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.markdown(
            """
<div class="glass-card">
  <div class="welcome-title">Welcome, Recruiter / Guest 👋</div>
  <div class="welcome-text">
    I’m <b>PersonaQuery</b> — Rudra’s digital twin. Ask about skills, projects, achievements,
    publications, and experience. I’ll answer using evidence from the provided documents.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="prompts-wrap">', unsafe_allow_html=True)

        prompt_cols = st.columns(2, gap="medium")
        prompts = [
            "Top 3 GenAI projects",
            "Best-fit roles",
            "Key strengths (ATS)",
            "Publications & patent",
        ]

        for i, p in enumerate(prompts):
            with prompt_cols[i % 2]:
                if st.button(p, key=f"prompt_{i}"):
                    st.session_state.user_input = p  # prefill input
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Chat panel (only after first message)
# -----------------------------
if has_chat:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    for m in st.session_state.messages:
        role = m["role"]
        content = m["content"]

        if role == "user":
            st.markdown(
                f"""
<div class="msg-row msg-user">
  <div class="bubble user">{content}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="msg-row msg-assistant">
  <div class="bubble assistant">{content}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Bottom input (always visible) — Enter works via st.form
# -----------------------------
st.markdown('<div class="pq-inputbar"><div class="pq-input-inner">', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=False):
    st.text_input("Message", key="user_input", placeholder="Type your question…")
    # label hidden by CSS, we use icon
    st.form_submit_button("➤", on_click=submit_message)

st.markdown("</div></div>", unsafe_allow_html=True)

# -----------------------------
# (Optional) tiny dev footer (invisible unless needed)
# -----------------------------
# st.caption(f"API: {st.session_state.api_url}")
