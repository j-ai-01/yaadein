import sqlite3
from yaadein.schema import connect, migrate, schema_version


def _tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] for r in rows}


def test_connect_creates_parent_dirs_and_db(tmp_path):
    db_path = tmp_path / "nested" / "memories.db"
    conn = connect(db_path)
    assert db_path.exists()
    conn.close()


def test_migrate_creates_tables_and_sets_version(tmp_path):
    conn = connect(tmp_path / "memories.db")
    migrate(conn)
    assert {"memories", "audit_log", "schema_version"} <= _tables(conn)
    assert schema_version(conn) == 2
    conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "memories.db")
    migrate(conn)
    migrate(conn)
    assert schema_version(conn) == 2
    conn.close()


def test_category_check_constraint_rejects_bad_value(tmp_path):
    conn = connect(tmp_path / "memories.db")
    migrate(conn)
    import pytest
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memories (id, content, category, scope_type, scope_key, created_at) "
            "VALUES ('m1', 'x', 'nonsense', 'user', '*', '2026-07-03T00:00:00')"
        )
    conn.close()
