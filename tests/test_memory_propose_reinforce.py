import math
import pytest
from yaadein.service import MemoryService
from yaadein.store import MemoryStore
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
        raise RuntimeError("ollama down")


def make_service(tmp_path, embedder=None):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=embedder or FakeEmbedder(),
        collection_name="test_memories",
    )
    return MemoryService(store=store, vector_index=index), store


def test_propose_lands_proposed_with_provenance(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.propose(
        content="User prefers pytest for testing",
        category="preference", scope_type="user", scope_key="*",
        confidence=0.8, evidence="I prefer pytest",
        source_harness="claude-code", source_session="sess-1",
    )
    row = store.get(saved.id)
    assert row.status == "proposed"
    assert row.confidence == 0.8
    assert row.evidence == "I prefer pytest"
    assert row.source_session == "sess-1"


def test_propose_rolls_back_on_embed_failure(tmp_path):
    service, store = make_service(tmp_path, embedder=ExplodingEmbedder())
    with pytest.raises(RuntimeError):
        service.propose(
            content="User prefers pytest for testing",
            category="preference", scope_type="user", scope_key="*",
            confidence=0.8,
        )
    assert store.list() == []


def test_find_similar_matches_same_scope_only(tmp_path):
    service, _ = make_service(tmp_path)
    service.propose(
        content="Deploys use the blue pipeline", category="fact",
        scope_type="project", scope_key="repo-a", confidence=0.9,
    )
    hit = service.find_similar("deploy pipeline colour", "project", "repo-a")
    assert hit is not None and hit[0].content == "Deploys use the blue pipeline"
    assert service.find_similar("deploy pipeline colour", "project", "repo-b") is None


def test_reinforce_bumps_confidence_capped_and_audited(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.propose(
        content="User prefers pytest for testing", category="preference",
        scope_type="user", scope_key="*", confidence=0.95,
    )
    service.reinforce(saved.id, source_session="sess-2")
    assert store.get(saved.id).confidence == 1.0
    actions = [row["action"] for row in store.audit_entries()]
    assert "reinforce" in actions
