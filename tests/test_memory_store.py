from yaadein.store import MemoryStore
from yaadein.types import Memory


def make_store(tmp_path):
    return MemoryStore(tmp_path / "memories.db")


def make_memory(**overrides):
    base = dict(
        id="", content="User prefers pytest over unittest",
        category="preference", scope_type="user", scope_key="*",
    )
    base.update(overrides)
    return Memory(**base)


def test_add_assigns_id_and_created_at(tmp_path):
    store = make_store(tmp_path)
    saved = store.add(make_memory())
    assert saved.id
    assert saved.created_at
    assert store.get(saved.id).content == "User prefers pytest over unittest"


def test_get_unknown_id_returns_none(tmp_path):
    store = make_store(tmp_path)
    assert store.get("nope") is None


def test_list_filters_by_scope_and_status(tmp_path):
    store = make_store(tmp_path)
    store.add(make_memory())
    proj = store.add(make_memory(
        content="Deploys go through pipeline X",
        category="fact", scope_type="project", scope_key="repo-a",
        status="confirmed",
    ))
    result = store.list(scope_type="project", scope_key="repo-a", status="confirmed")
    assert [m.id for m in result] == [proj.id]


def test_record_retrieval_bumps_counter_and_timestamp(tmp_path):
    store = make_store(tmp_path)
    saved = store.add(make_memory())
    store.record_retrieval([saved.id])
    store.record_retrieval([saved.id])
    fetched = store.get(saved.id)
    assert fetched.times_retrieved == 2
    assert fetched.last_retrieved is not None


def test_set_status_updates_row(tmp_path):
    store = make_store(tmp_path)
    saved = store.add(make_memory())
    store.set_status(saved.id, "confirmed")
    assert store.get(saved.id).status == "confirmed"


def test_forget_deletes_and_reports(tmp_path):
    store = make_store(tmp_path)
    saved = store.add(make_memory())
    assert store.forget(saved.id) is True
    assert store.get(saved.id) is None
    assert store.forget(saved.id) is False


def test_every_mutation_is_audited(tmp_path):
    store = make_store(tmp_path)
    saved = store.add(make_memory())
    store.record_retrieval([saved.id])
    store.set_status(saved.id, "confirmed")
    store.forget(saved.id)
    actions = [row["action"] for row in store.audit_entries()]
    assert actions == ["add", "retrieve", "status:confirmed", "forget"]
