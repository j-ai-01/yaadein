# tests/scenarios/test_s5_privacy_gates.py
# WHY: Privacy is a hard guarantee. Redacted content must never appear in any
# output path. Scope isolation must hold even when the vector index returns
# cross-scope results (the service layer is the last line of defense).

import pytest
from unittest.mock import MagicMock
from yaadein.store import MemoryStore
from yaadein.service import MemoryService


def _make_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MagicMock()
    index.query.return_value = []
    index.add = MagicMock()
    ep_index = MagicMock()
    ep_index.query.return_value = []
    ep_index.add = MagicMock()
    return store, MemoryService(store=store, vector_index=index, episode_index=ep_index)


def test_s5_redact_strips_api_keys():
    """redact() removes API key patterns from text."""
    from yaadein.redact import redact

    text = "My key is sk-abc123xyzabc123xyzabc123 and my token is ghp_ABCDEF1234567890abcdef"
    cleaned = redact(text)
    assert "sk-abc123xyzabc123xyzabc123" not in cleaned
    assert "ghp_ABCDEF" not in cleaned


def test_s5_forgotten_memory_not_recalled(tmp_path):
    """After forget, the memory does not appear in recall results."""
    store, service = _make_service(tmp_path)

    mem = service.propose(
        content="sensitive personal info",
        category="fact",
        scope_type="user",
        scope_key="*",
        confidence=0.9,
        source_session="priv-sess-001",
    )
    mem_id = mem.id

    assert service.forget(mem_id) is True

    # Wire index to return the (now-deleted) memory id
    service._index.query.return_value = [(mem_id, 0.95)]

    results = service.recall("sensitive personal info", project_key=None)
    assert all(r["id"] != mem_id for r in results), "Forgotten memory still returned by recall"


def test_s5_forgotten_memory_not_in_briefing(tmp_path):
    """After forget, the memory does not appear in briefing."""
    store, service = _make_service(tmp_path)

    mem = service.propose(
        content="confidential decision about architecture",
        category="decision",
        scope_type="user",
        scope_key="*",
        confidence=0.9,
        source_session="priv-sess-002",
    )
    mem_id = mem.id

    assert service.forget(mem_id) is True

    briefing = service.briefing(project_key=None)
    all_ids = [m["id"] for m in briefing.get("decisions", [])]
    assert mem_id not in all_ids, "Forgotten memory appears in briefing"


def test_s5_project_scope_isolation_in_briefing(tmp_path):
    """Project B's memories never appear in Project A's briefing."""
    store, service = _make_service(tmp_path)

    # Store a project-B-scoped memory
    service.propose(
        content="Project B secret architecture decision",
        category="decision",
        scope_type="project",
        scope_key="proj-b",
        confidence=0.9,
        source_session="priv-sess-003",
    )

    # Query briefing as project A
    briefing = service.briefing(project_key="proj-a")
    all_content = [
        m["content"]
        for category in ["facts", "decisions", "gotchas", "conflicts"]
        for m in briefing.get(category, [])
    ]
    assert all("Project B secret" not in c for c in all_content), \
        "Project B memory leaked into Project A briefing"


def test_s5_scope_isolation_survives_index_bleed(tmp_path):
    """Even if vector index returns cross-scope IDs, service filters them out."""
    store, service = _make_service(tmp_path)

    # Seed a project-B memory
    mem_b = service.propose(
        content="project B only fact",
        category="fact",
        scope_type="project",
        scope_key="proj-b",
        confidence=0.9,
        source_session="priv-sess-004",
    )
    mem_b_id = mem_b.id

    # Force the index to return project B's memory for a project A query
    service._index.query.return_value = [(mem_b_id, 0.99)]

    results = service.recall("project B only fact", project_key="proj-a")
    assert all(r["id"] != mem_b_id for r in results), \
        "Scope isolation failed: project B memory returned in project A recall"
