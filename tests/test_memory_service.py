import math
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from yaadein.service import MemoryService, get_memory_service
from yaadein.store import MemoryStore
from yaadein.types import Episode, Memory
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


def make_episode_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma", embedder=FakeEmbedder(),
        collection_name="test_memories",
    )
    episode_index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma_ep", embedder=FakeEmbedder(),
        collection_name="test_episodes",
    )
    return MemoryService(store=store, vector_index=index, episode_index=episode_index), store


def test_find_similar_skips_archived_and_superseded_and_wrong_scope(tmp_path):
    """find_similar's scan skips a missing memory, an archived one, a superseded one, and an out-of-scope match before finding the real hit."""
    service, store = make_service(tmp_path)
    archived = store.add(Memory(
        id="", content="Deploys go through pipeline auth", category="fact",
        scope_type="user", scope_key="*", status="archived",
    ))
    service._index.add(archived.id, archived.content)
    superseded = store.add(Memory(
        id="", content="Deploys go through pipeline auth", category="fact",
        scope_type="user", scope_key="*", status="confirmed",
        superseded_by="mem_newer",
    ))
    service._index.add(superseded.id, superseded.content)
    other_scope = store.add(Memory(
        id="", content="Deploys go through pipeline auth", category="fact",
        scope_type="project", scope_key="repo-b", status="confirmed",
    ))
    service._index.add(other_scope.id, other_scope.content)
    real = store.add(Memory(
        id="", content="Deploys go through pipeline auth", category="fact",
        scope_type="user", scope_key="*", status="confirmed",
    ))
    service._index.add(real.id, real.content)

    result = service.find_similar("Deploys go through pipeline auth", "user", "*")
    assert result is not None
    assert result[0].id == real.id


def test_recall_episodes_returns_empty_without_episode_index(tmp_path):
    """recall_episodes returns [] immediately when no episode index is configured."""
    service, _ = make_service(tmp_path)
    assert service.recall_episodes("anything") == []


def test_recall_episodes_skips_missing_episode_and_out_of_scope(tmp_path):
    """recall_episodes skips a query hit whose episode row is gone and one out of scope."""
    service, store = make_episode_service(tmp_path)
    service._episode_index.add("ep_ghost", "ghost summary about kyun")
    ep = service.record_episode(
        summary="Designed Kyun provenance and auth flow.",
        excerpt="USER: kyun idea...",
        scope_type="project", scope_key="repo-a",
    )
    hits_wrong_scope = service.recall_episodes("kyun provenance", project_key="repo-b")
    assert hits_wrong_scope == []
    hits_right_scope = service.recall_episodes("kyun provenance", project_key="repo-a")
    assert hits_right_scope[0]["id"] == ep.id


def test_recall_episodes_handles_bad_created_at(tmp_path):
    """A malformed created_at on an episode falls back to age_days=0 instead of raising."""
    service, store = make_episode_service(tmp_path)
    ep = Episode(
        id="", scope_type="user", scope_key="*",
        summary="Designed Kyun provenance.", excerpt="excerpt",
        created_at="not-a-timestamp",
    )
    saved = store.add_episode(ep)
    service._episode_index.add(saved.id, saved.summary)
    hits = service.recall_episodes("kyun provenance")
    assert hits[0]["id"] == saved.id


def test_briefing_skips_out_of_scope_episode(tmp_path):
    """briefing's recent_conversations section skips episodes outside the requested project scope."""
    service, store = make_episode_service(tmp_path)
    service.record_episode(
        summary="Project repo-a work.", excerpt="excerpt",
        scope_type="project", scope_key="repo-a",
    )
    briefing = service.briefing(project_key="repo-b")
    assert briefing["recent_conversations"] == []


def test_in_scope_pair_shared_scope_is_not_visible():
    """A scope_type outside {user, project} (e.g. future 'shared') is not visible anywhere yet."""
    assert MemoryService._in_scope_pair("shared", "team-x", "team-x") is False


def test_get_memory_service_builds_singleton_without_network(monkeypatch):
    """get_memory_service constructs the singleton via OllamaEmbedder, mocked here to avoid any network call."""
    import yaadein.service as service_module
    monkeypatch.setattr(service_module, "_service", None)

    fake_embedder_cls = MagicMock()
    fake_module = types.ModuleType("yaadein.vector_index")
    fake_module.OllamaEmbedder = fake_embedder_cls
    fake_module.MemoryVectorIndex = MagicMock()

    with patch.dict(sys.modules, {"yaadein.vector_index": fake_module}):
        with patch("yaadein.store.MemoryStore"):
            result = service_module.get_memory_service()
            again = service_module.get_memory_service()

    assert result is again
    monkeypatch.setattr(service_module, "_service", None)
