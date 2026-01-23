import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PersonaQuery", page_icon="🧠", layout="centered")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("PersonaQuery 🧠")
st.caption("Ask questions about Rudra’s resume, patent, and research paper (RAG + Groq).")

# Session chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    st.write("Tip: use Render URL after deployment.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(
                        f"- **{s.get('file_name','unknown')}** (page {s.get('page_label','?')}) — score `{s.get('score',0):.3f}`"
                    )
                    st.caption(s.get("snippet", ""))

# Input
question = st.chat_input("Ask PersonaQuery something…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{api_url}/chat",
                    json={"question": question},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
            except Exception as e:
                answer = f"❌ Error calling API: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"- **{s.get('file_name','unknown')}** (page {s.get('page_label','?')}) — score `{s.get('score',0):.3f}`"
                    )
                    st.caption(s.get("snippet", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
