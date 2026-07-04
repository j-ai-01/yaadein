import math

import pytest

from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.types import Memory
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class ExplodingEmbedder:
    def embed(self, text):
        raise RuntimeError("embedding backend unavailable")


def make_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_memories",
    )
    return MemoryService(store=store, vector_index=index), store


def test_remember_lands_confirmed(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest", category="preference")
    assert store.get(saved.id).status == "confirmed"


def test_recall_returns_scored_dicts_ranked(tmp_path):
    service, _ = make_service(tmp_path)
    service.remember("User prefers pytest for testing", category="preference")
    service.remember("Deploys go through the blue pipeline", category="fact")
    results = service.recall("which pytest testing setup?")
    assert results[0]["content"] == "User prefers pytest for testing"
    assert results[0]["score"] > results[1]["score"]


def test_recall_excludes_archived(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    store.set_status(saved.id, "archived")
    assert service.recall("pytest") == []


def test_recall_scopes_project_memories(tmp_path):
    service, _ = make_service(tmp_path)
    service.remember(
        "Auth module is fragile", category="gotcha",
        scope_type="project", scope_key="repo-a",
    )
    with_project = service.recall("auth fragile?", project_key="repo-a")
    without_project = service.recall("auth fragile?", project_key="repo-b")
    assert len(with_project) == 1
    assert without_project == []


def test_recall_records_retrieval(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    service.recall("pytest")
    assert store.get(saved.id).times_retrieved == 1


def test_forget_removes_everywhere(tmp_path):
    service, _ = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    assert service.forget(saved.id) is True
    assert service.recall("pytest") == []


def test_remember_rolls_back_store_on_embed_failure(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=ExplodingEmbedder(),
        collection_name="test_memories",
    )
    service = MemoryService(store=store, vector_index=index)
    with pytest.raises(RuntimeError):
        service.remember("User prefers pytest for testing")
    assert store.list() == []


def test_recall_excludes_superseded(tmp_path):
    service, store = make_service(tmp_path)
    superseded = Memory(
        id="", content="User prefers pytest for testing", category="preference",
        scope_type="user", scope_key="*", status="confirmed",
        superseded_by="mem_newer",
    )
    saved = store.add(superseded)
    index_used = service._index
    index_used.add(saved.id, saved.content)
    assert service.recall("pytest") == []
