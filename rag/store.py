"""ChromaDB vector store wrapper with singleton client for reuse."""

import json
import os
import threading
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

_store_lock = threading.Lock()
_store: tuple | None = None


def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_store():
    """Return (collection, cfg). Client is created once and reused across calls."""
    global _store
    if _store is not None:
        return _store

    with _store_lock:
        if _store is not None:
            return _store

        cfg = _load_config()["rag"]
        persist_dir = os.path.expanduser(cfg["persist_dir"])
        os.makedirs(persist_dir, exist_ok=True)

        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg["embedding_model"],
        )

        collection = client.get_or_create_collection(
            name=cfg["collection_name"],
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        _store = (collection, cfg)
        return _store


def reset_store():
    """Delete and recreate the collection (for re-indexing)."""
    global _store
    cfg = _load_config()["rag"]
    persist_dir = os.path.expanduser(cfg["persist_dir"])
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(cfg["collection_name"])
    except Exception:
        pass
    _store = None
    print("Collection deleted. Run ingest again to re-index.")
