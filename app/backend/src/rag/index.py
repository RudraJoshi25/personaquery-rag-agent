from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from app.backend.src.core.config import INDEX_PERSIST_DIR
from app.backend.src.rag.ingest import load_documents


def build_index(private_data_dir: str):
    # Chunking: keep it simple for MVP; we’ll improve with “section-aware” splitting next
    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=150)

    docs = load_documents(private_data_dir)
    nodes = splitter.get_nodes_from_documents(docs)

    # FAISS vector store
    dim = 1536  # text-embedding-3-small = 1536 dims
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
    return load_index_from_storage(storage_context)
