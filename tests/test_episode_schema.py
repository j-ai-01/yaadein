from yaadein.schema import MIGRATIONS, connect, migrate, schema_version


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def _memory_columns(conn):
    return {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}


def test_fresh_db_migrates_to_v2_with_episodes(tmp_path):
    conn = connect(tmp_path / "m.db")
    migrate(conn)
    assert schema_version(conn) == 2
    assert "episodes" in _tables(conn)
    assert "episode_id" in _memory_columns(conn)
    conn.close()


def test_v1_db_migrates_losslessly(tmp_path):
    conn = connect(tmp_path / "m.db")
    # build a genuine v1 database: apply only the first migration
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    conn.executescript(MIGRATIONS[0])
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute(
        "INSERT INTO memories (id, content, category, scope_type, scope_key, created_at) "
        "VALUES ('mem_v1row', 'old fact', 'fact', 'user', '*', '2026-07-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO audit_log (ts, action, memory_id, detail) "
        "VALUES ('2026-07-01T00:00:00', 'add', 'mem_v1row', NULL)"
    )
    conn.commit()

    migrate(conn)
    assert schema_version(conn) == 2
    row = conn.execute("SELECT * FROM memories WHERE id='mem_v1row'").fetchone()
    assert row["content"] == "old fact"
    assert row["episode_id"] is None
    assert conn.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"] == 1
    conn.close()


def test_episode_scope_check_constraint(tmp_path):
    import pytest, sqlite3
    conn = connect(tmp_path / "m.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO episodes (id, scope_type, scope_key, summary, excerpt, created_at) "
            "VALUES ('ep_x', 'galaxy', '*', 's', 'e', '2026-07-05T00:00:00')"
        )
    conn.close()
