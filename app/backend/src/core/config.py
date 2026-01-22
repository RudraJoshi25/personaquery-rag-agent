import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "storage")
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

AUTHOR_LINKS = {
    "HealthEcho": "https://healthecho.vercel.app/",
    "GitHub": "https://github.com/RudraJoshi25",
    "LinkedIn": "https://linkedin.com/in/rudrajoshi25",
    "Research Paper (IEEE Xplore)": "https://ieeexplore.ieee.org/document/10065338",
}
