from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.types import Memory
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    def embed(self, text):
        return [0.5, 0.5, 0.5, 0.5]


def make_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_memories",
    )
    return MemoryService(store=store, vector_index=index), store


def test_briefing_groups_by_category(tmp_path):
    service, _ = make_service(tmp_path)
    service.remember("User prefers pytest", category="preference")
    service.remember("Rejected GraphQL in March", category="decision")
    service.remember(
        "Auth module is fragile", category="gotcha",
        scope_type="project", scope_key="repo-a",
    )
    result = service.briefing(project_key="repo-a")
    assert [m["content"] for m in result["facts"]] == ["User prefers pytest"]
    assert [m["content"] for m in result["decisions"]] == ["Rejected GraphQL in March"]
    assert [m["content"] for m in result["gotchas"]] == ["Auth module is fragile"]
    assert result["conflicts"] == []


def test_briefing_excludes_other_projects_and_archived(tmp_path):
    service, store = make_service(tmp_path)
    service.remember(
        "Other repo gotcha", category="gotcha",
        scope_type="project", scope_key="repo-b",
    )
    archived = service.remember("Old preference", category="preference")
    store.set_status(archived.id, "archived")
    result = service.briefing(project_key="repo-a")
    assert result["facts"] == []
    assert result["gotchas"] == []


def test_briefing_marks_proposed_gotchas_unconfirmed(tmp_path):
    service, store = make_service(tmp_path)
    proposed = Memory(
        id="", content="Flaky test in CI?", category="gotcha",
        scope_type="user", scope_key="*", status="proposed",
    )
    store.add(proposed)
    result = service.briefing()
    assert result["gotchas"][0]["unconfirmed"] is True


def test_briefing_surfaces_conflicts(tmp_path):
    service, store = make_service(tmp_path)
    a = service.remember("Prefers unittest", category="preference")
    conflicted = Memory(
        id="", content="Prefers pytest now", category="preference",
        scope_type="user", scope_key="*", status="proposed",
        conflict_with=a.id,
    )
    store.add(conflicted)
    result = service.briefing()
    assert [m["content"] for m in result["conflicts"]] == ["Prefers pytest now"]


def test_briefing_records_retrieval(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest", category="preference")
    service.briefing()
    assert store.get(saved.id).times_retrieved == 1


def test_briefing_dedupes_retrieval_for_overlapping_sections(tmp_path):
    """A confirmed preference with a conflict should appear in both facts and conflicts,
    but times_retrieved should only be incremented once."""
    service, store = make_service(tmp_path)
    # Create a confirmed preference with a conflict_with set
    confirmed_with_conflict = Memory(
        id="", content="Prefers pytest", category="preference",
        scope_type="user", scope_key="*", status="confirmed",
        conflict_with="some-other-id",
    )
    saved = store.add(confirmed_with_conflict)
    service.briefing()
    assert store.get(saved.id).times_retrieved == 1
