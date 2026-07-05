import math
import sys
import types
from unittest.mock import MagicMock, patch

from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    """Deterministic 4-dim embeddings keyed on which words appear."""

    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_index(tmp_path):
    return MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_memories",
    )


def test_query_ranks_semantically_closest_first(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    index.add("m2", "Deploys go through the blue pipeline")
    results = index.query("what testing framework? pytest?", top_k=2)
    assert results[0][0] == "m1"
    assert results[0][1] > results[1][1]


def test_scores_are_similarities_between_zero_and_one(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    results = index.query("pytest testing", top_k=1)
    assert 0.0 <= results[0][1] <= 1.0


def test_delete_removes_from_results(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    index.delete("m1")
    assert index.query("pytest", top_k=1) == []


def test_ollama_embedder_wraps_local_model_without_network():
    """OllamaEmbedder builds an OllamaEmbedding client and delegates embed() to it, with no real network call."""
    fake_ollama_embedding_cls = MagicMock()
    fake_instance = fake_ollama_embedding_cls.return_value
    fake_instance.get_text_embedding.return_value = [0.1, 0.2, 0.3]

    fake_module = types.ModuleType("llama_index.embeddings.ollama")
    fake_module.OllamaEmbedding = fake_ollama_embedding_cls

    with patch.dict(sys.modules, {"llama_index.embeddings.ollama": fake_module}):
        from yaadein.vector_index import OllamaEmbedder

        embedder = OllamaEmbedder()
        result = embedder.embed("hello world")

    fake_ollama_embedding_cls.assert_called_once()
    fake_instance.get_text_embedding.assert_called_once_with("hello world")
    assert result == [0.1, 0.2, 0.3]
