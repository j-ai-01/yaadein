# tests/test_episode_service.py
import math
from datetime import datetime, timedelta, timezone

import pytest

from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.types import Episode
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["kyun", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class ExplodingEmbedder:
    def embed(self, text):
        raise RuntimeError("down")


def make_service(tmp_path, episode_embedder=None):
    store = MemoryStore(tmp_path / "m.db")
    facts = MemoryVectorIndex(tmp_path / "cf", FakeEmbedder(), "t_facts")
    episodes = MemoryVectorIndex(tmp_path / "ce", episode_embedder or FakeEmbedder(), "t_eps")
    return MemoryService(store=store, vector_index=facts, episode_index=episodes), store


def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_record_episode_persists_and_is_recallable(tmp_path):
    service, store = make_service(tmp_path)
    ep = service.record_episode(
        summary="Designed the Kyun provenance project.",
        excerpt="USER: kyun idea...", scope_type="project", scope_key="repo-a",
        session_id="sess-1",
    )
    assert store.get_episode(ep.id) is not None
    hits = service.recall_episodes("what did we discuss about kyun?", project_key="repo-a")
    assert hits[0]["id"] == ep.id


def test_record_episode_rolls_back_on_index_failure(tmp_path):
    service, store = make_service(tmp_path, episode_embedder=ExplodingEmbedder())
    with pytest.raises(RuntimeError):
        service.record_episode(
            summary="Kyun things", excerpt="x", scope_type="user", scope_key="*",
        )
    assert store.list_episodes() == []


def test_recall_returns_trimmed_payload_without_excerpt(tmp_path):
    service, _ = make_service(tmp_path)
    service.record_episode(
        summary="Designed the Kyun provenance project.",
        excerpt="USER: kyun idea..." * 100, scope_type="user", scope_key="*",
        session_id="sess-1",
    )
    hits = service.recall_episodes("kyun")
    assert "excerpt" not in hits[0]
    assert set(hits[0]) == {
        "id", "summary", "created_at", "session_id",
        "scope_type", "scope_key", "score",
    }


def test_recall_scope_filtering(tmp_path):
    service, _ = make_service(tmp_path)
    service.record_episode(summary="Kyun repo-a talk", excerpt="x",
                           scope_type="project", scope_key="repo-a")
    assert service.recall_episodes("kyun", project_key="repo-b") == []
    assert len(service.recall_episodes("kyun", project_key="repo-a")) == 1


def test_recency_breaks_similarity_ties(tmp_path):
    service, store = make_service(tmp_path)
    old = store.add_episode(Episode(id="", scope_type="user", scope_key="*",
                                    summary="kyun talk one", excerpt="x",
                                    created_at=_iso(days_ago=30)))
    new = store.add_episode(Episode(id="", scope_type="user", scope_key="*",
                                    summary="kyun talk one", excerpt="x",
                                    created_at=_iso(days_ago=0)))
    service._episode_index.add(old.id, old.summary)
    service._episode_index.add(new.id, new.summary)
    hits = service.recall_episodes("kyun talk")
    assert hits[0]["id"] == new.id  # identical similarity; recency decides


def test_read_episode_includes_fact_ids_and_handles_unknown(tmp_path):
    service, store = make_service(tmp_path)
    ep = service.record_episode(summary="Kyun design", excerpt="x",
                                scope_type="user", scope_key="*")
    fact = service.propose(content="Kyun uses content matching for provenance",
                           category="decision", scope_type="user", scope_key="*",
                           confidence=0.9, episode_id=ep.id)
    detail = service.read_episode(ep.id)
    assert detail["fact_ids"] == [fact.id]
    assert service.read_episode("ep_unknown") is None


def test_briefing_lists_recent_conversations(tmp_path):
    service, _ = make_service(tmp_path)
    for i in range(4):
        service.record_episode(summary=f"Talk number {i}. More detail here.",
                               excerpt="x", scope_type="user", scope_key="*")
    briefing = service.briefing()
    convs = briefing["recent_conversations"]
    assert len(convs) == 3  # capped by MEMORY_BRIEFING_LIMITS["conversations"]
    assert convs[0]["summary"] == "Talk number 3"  # first sentence only


def test_no_episode_index_degrades_gracefully(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    facts = MemoryVectorIndex(tmp_path / "cf", FakeEmbedder(), "t_facts")
    service = MemoryService(store=store, vector_index=facts)
    assert service.has_episode_index is False
    assert service.recall_episodes("kyun") == []
    assert service.briefing()["recent_conversations"] == []
    with pytest.raises(RuntimeError):
        service.record_episode(summary="s", excerpt="e",
                               scope_type="user", scope_key="*")
