# app/backend/src/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Where the index is stored locally (ignored by git via .gitignore: storage/)
INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "storage")

# Where your private PDFs live (ignored by git)
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")



AUTHOR_LINKS = {
    "HealthEcho": "https://healthecho.vercel.app/",
    "GitHub": "https://github.com/RudraJoshi25",
    "LinkedIn": "https://linkedin.com/in/rudrajoshi25",
    "Research Paper (IEEE Xplore)": "https://ieeexplore.ieee.org/document/10065338",
}
