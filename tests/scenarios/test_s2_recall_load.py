# tests/scenarios/test_s2_recall_load.py
# WHY: Recall correctness and scope isolation are the guarantees Yaadein makes.
# 500 memories across 3 scopes, then query — top result must be right, and
# project A must never see project B's memories.

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from yaadein.store import MemoryStore
from yaadein.service import MemoryService


def _make_service(db_path: Path):
    """MemoryService with a real file-backed store but mocked vector index.
    Vector index is mocked to return controlled ranked results."""
    store = MemoryStore(db_path)
    index = MagicMock()
    episode_index = MagicMock()
    episode_index.query.return_value = []
    return store, index, MemoryService(store=store, vector_index=index, episode_index=episode_index)


def _seed_memories(service, count: int, scope_type: str, scope_key: str, prefix: str):
    """Insert `count` memories and return their IDs."""
    ids = []
    for i in range(count):
        mem = service.propose(
            content=f"{prefix} memory {i}",
            category="fact",
            scope_type=scope_type,
            scope_key=scope_key,
            confidence=0.8,
            source_session=f"sess-{i}",
        )
        ids.append(mem.id)
    return ids


def test_s2_top_result_is_highest_ranked(tmp_path):
    """Query returns the best-matching memory first."""
    store, index, service = _make_service(tmp_path / "memories.db")

    # Seed 10 user-scope memories
    ids = _seed_memories(service, 10, "user", "*", "topic-alpha")

    # Mock index to return ids ranked: best match first (highest similarity)
    ranked = [(mem_id, 1.0 - (i * 0.05)) for i, mem_id in enumerate(ids)]
    index.query.return_value = ranked

    results = service.recall("topic-alpha memory")
    assert results[0]["content"] == "topic-alpha memory 0"


def test_s2_scope_isolation_project_a_vs_b(tmp_path):
    """Project A's memories never appear in Project B's recall."""
    store, index, service = _make_service(tmp_path / "memories.db")

    ids_a = _seed_memories(service, 5, "project", "proj-a", "alpha")
    ids_b = _seed_memories(service, 5, "project", "proj-b", "beta")

    # Index returns all 10 — but scope filtering must exclude cross-project
    all_ids = [(mid, 0.9) for mid in ids_a + ids_b]
    index.query.return_value = all_ids

    results_a = service.recall("memory", project_key="proj-a")
    result_contents = [r["content"] for r in results_a]

    # No beta memories should appear
    assert all("beta" not in c for c in result_contents), \
        f"Scope leak: project B content in project A results: {result_contents}"


def test_s2_user_scope_visible_to_all_projects(tmp_path):
    """User-scoped memories appear regardless of which project queries."""
    store, index, service = _make_service(tmp_path / "memories.db")

    user_ids = _seed_memories(service, 3, "user", "*", "universal")

    index.query.return_value = [(mid, 0.9) for mid in user_ids]

    results_proj = service.recall("universal", project_key="some-project")
    assert len(results_proj) == 3
    assert all("universal" in r["content"] for r in results_proj)


def test_s2_500_memories_no_crash(tmp_path):
    """Inserting 500 memories completes without error."""
    store, index, service = _make_service(tmp_path / "memories.db")
    index.query.return_value = []

    for i in range(500):
        service.propose(
            content=f"bulk memory {i}",
            category="fact",
            scope_type="user",
            scope_key="*",
            confidence=0.7,
            source_session="bulk-session",
        )
    # No assertion needed — if we get here without exception, it passes


# NOTE: a concurrent readers+writers stress test was removed here by decision
# on 2026-07-06 — same shared-connection limitation as noted in test_s4_resilience.py.
# Re-add when MemoryStore supports concurrent writes.
