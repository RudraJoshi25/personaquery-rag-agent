from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from app.backend.src.core.config import OPENAI_API_KEY, OPENAI_MODEL, EMBED_MODEL
from app.backend.src.rag.index import load_index


def get_query_engine():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env")

    Settings.llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    index = load_index()

    # Response mode keeps citations/source nodes accessible
    return index.as_query_engine(similarity_top_k=5, response_mode="compact")
