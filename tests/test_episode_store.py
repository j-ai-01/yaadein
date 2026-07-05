from yaadein.store import MemoryStore
from yaadein.types import Episode, Memory


def make_store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def make_episode(**overrides):
    base = dict(
        id="", scope_type="project", scope_key="repo-a",
        summary="Discussed the Kyun design.", excerpt="USER: kyun...",
        session_id="sess-1",
    )
    base.update(overrides)
    return Episode(**base)


def test_add_assigns_id_and_created_at(tmp_path):
    store = make_store(tmp_path)
    saved = store.add_episode(make_episode())
    assert saved.id.startswith("ep_")
    assert saved.created_at
    assert store.get_episode(saved.id).summary == "Discussed the Kyun design."


def test_add_honors_preset_id(tmp_path):
    store = make_store(tmp_path)
    saved = store.add_episode(make_episode(id="ep_preset000001"))
    assert saved.id == "ep_preset000001"
    assert store.get_episode("ep_preset000001") is not None


def test_get_unknown_returns_none(tmp_path):
    assert make_store(tmp_path).get_episode("ep_nope") is None


def test_list_newest_first_with_limit_and_scope(tmp_path):
    store = make_store(tmp_path)
    store.add_episode(make_episode(created_at="2026-07-01T00:00:00+00:00"))
    newest = store.add_episode(make_episode(created_at="2026-07-03T00:00:00+00:00"))
    store.add_episode(make_episode(scope_key="repo-b", created_at="2026-07-05T00:00:00+00:00"))
    result = store.list_episodes(scope_type="project", scope_key="repo-a", limit=1)
    assert [e.id for e in result] == [newest.id]


def test_fact_ids_for_episode(tmp_path):
    store = make_store(tmp_path)
    ep = store.add_episode(make_episode())
    m1 = store.add(Memory(id="", content="fact one about kyun", category="fact",
                          scope_type="project", scope_key="repo-a", episode_id=ep.id))
    store.add(Memory(id="", content="unrelated", category="fact",
                     scope_type="user", scope_key="*"))
    assert store.fact_ids_for_episode(ep.id) == [m1.id]


def test_delete_episode_and_audit_trail(tmp_path):
    store = make_store(tmp_path)
    ep = store.add_episode(make_episode())
    store.delete_episode(ep.id)
    assert store.get_episode(ep.id) is None
    actions = [r["action"] for r in store.audit_entries()]
    assert actions == ["add_episode", "rollback_episode"]
