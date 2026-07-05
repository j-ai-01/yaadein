"""The Chroma-backed semantic index over memory content.

Chroma holds only per-memory embeddings, keyed by the same id SQLite uses for
the authoritative record — "Chroma nominates, SQLite decides." This module
never stores or judges memory state; it only ranks by similarity for recall.
"""

from pathlib import Path
from typing import List, Protocol, Tuple

from utils.chroma_client import make_chroma_client


class Embedder(Protocol):
    """Anything that can turn text into a fixed-size embedding vector."""

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for `text` (tests inject fakes here)."""
        ...


class OllamaEmbedder:
    """Embedder backed by a local Ollama embedding model (see config.EMBED_MODEL)."""

    def __init__(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from config import EMBED_MODEL, OLLAMA_BASE_URL

        self._model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def embed(self, text: str) -> List[float]:
        """Embed `text` via the local Ollama model (one network call to localhost)."""
        return self._model.get_text_embedding(text)


class MemoryVectorIndex:
    """Thin wrapper over a Chroma collection: embed, upsert, similarity-query,
    and delete by memory id. Holds no opinion about memory status or scope —
    callers (MemoryService) filter results against the SQLite store."""

    def __init__(self, chroma_dir: Path, embedder: Embedder, collection_name: str):
        self._embedder = embedder
        client = make_chroma_client(str(chroma_dir))
        self._collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, memory_id: str, content: str) -> None:
        """Embed and upsert a memory's content into the collection, keyed by memory_id."""
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[self._embedder.embed(content)],
            documents=[content],
        )

    def query(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Return up to top_k (memory_id, similarity) pairs nearest to `text`,
        similarity in [0, 1] (1 = identical). Empty list if the collection is empty."""
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed(text)],
            n_results=min(top_k, count),
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # cosine distance -> similarity, clamped to [0, 1]
        return [
            (memory_id, max(0.0, min(1.0, 1.0 - dist)))
            for memory_id, dist in zip(ids, distances)
        ]

    def delete(self, memory_id: str) -> None:
        """Remove a memory's embedding from the collection."""
        self._collection.delete(ids=[memory_id])
