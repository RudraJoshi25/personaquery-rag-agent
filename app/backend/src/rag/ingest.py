from pathlib import Path
from llama_index.core import SimpleDirectoryReader


def load_documents(private_data_dir: str):
    data_path = Path(private_data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"PRIVATE_DATA_DIR not found: {data_path}")

    # Reads PDFs + other files in folder
    reader = SimpleDirectoryReader(input_dir=str(data_path), recursive=True)
    return reader.load_data()
