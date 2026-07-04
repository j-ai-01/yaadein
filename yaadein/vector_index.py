from pathlib import Path
from typing import List, Protocol, Tuple

from utils.chroma_client import make_chroma_client


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class OllamaEmbedder:
    def __init__(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from config import EMBED_MODEL, OLLAMA_BASE_URL

        self._model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def embed(self, text: str) -> List[float]:
        return self._model.get_text_embedding(text)


class MemoryVectorIndex:
    def __init__(self, chroma_dir: Path, embedder: Embedder, collection_name: str):
        self._embedder = embedder
        client = make_chroma_client(str(chroma_dir))
        self._collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, memory_id: str, content: str) -> None:
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[self._embedder.embed(content)],
            documents=[content],
        )

    def query(self, text: str, top_k: int) -> List[Tuple[str, float]]:
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
        self._collection.delete(ids=[memory_id])
