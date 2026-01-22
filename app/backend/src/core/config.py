import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "storage")
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
