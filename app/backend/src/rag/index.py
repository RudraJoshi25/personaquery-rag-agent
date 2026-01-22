from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from src.core.config import INDEX_PERSIST_DIR
from src.rag.ingest import load_documents


def build_index(private_data_dir: str):
    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=150)

    docs = load_documents(private_data_dir)
    nodes = splitter.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes)

    Path(INDEX_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
    return load_index_from_storage(storage_context)
