"""SQLite schema and migrations for the memory store.

SQLite is the source of truth for memories: content, category, scope, status
(proposed -> confirmed -> archived), confidence, provenance, and the audit
log. Chroma (see vector_index.py) only holds embeddings alongside the same
ids for semantic search; SQLite decides what a memory means and is.
"""

import sqlite3
from pathlib import Path

MIGRATIONS = [
    # v1: initial schema
    """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL
            CHECK (category IN ('preference', 'decision', 'fact', 'gotcha')),
        scope_type TEXT NOT NULL
            CHECK (scope_type IN ('user', 'project', 'shared')),
        scope_key TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'proposed'
            CHECK (status IN ('proposed', 'confirmed', 'archived')),
        confidence REAL NOT NULL DEFAULT 1.0,
        source_harness TEXT,
        source_session TEXT,
        evidence TEXT,
        created_at TEXT NOT NULL,
        last_retrieved TEXT,
        times_retrieved INTEGER NOT NULL DEFAULT 0,
        times_used INTEGER NOT NULL DEFAULT 0,
        superseded_by TEXT,
        conflict_with TEXT
    );

    CREATE TABLE audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        action TEXT NOT NULL,
        memory_id TEXT,
        detail TEXT
    );

    CREATE INDEX idx_memories_scope ON memories (scope_type, scope_key);
    CREATE INDEX idx_memories_status ON memories (status);
    """,
    # v2: episodic memory — episodes table + fact->episode link (design D1)
    """
    CREATE TABLE episodes (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        source_harness TEXT,
        scope_type TEXT NOT NULL
            CHECK (scope_type IN ('user', 'project', 'shared')),
        scope_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        excerpt TEXT NOT NULL,
        transcript_path TEXT,
        transcript_format TEXT,
        turn_start INTEGER,
        turn_end INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_episodes_scope ON episodes (scope_type, scope_key);
    CREATE INDEX idx_episodes_created ON episodes (created_at);

    ALTER TABLE memories ADD COLUMN episode_id TEXT;
    CREATE INDEX idx_memories_episode ON memories (episode_id);
    """,
]


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating parent dirs as needed) the shared SQLite connection used
    by the whole process, with row access by column name."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: this connection is shared as a singleton and used
    # from background threadpool threads (e.g. extraction). CPython's sqlite3
    # module is built against SQLite compiled with serialized threading mode
    # (sqlite3.threadsafety == 3), so sharing one connection across threads is safe.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """Return the currently applied migration index, or 0 if none have run yet."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return row["version"] if row else 0


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any migrations in MIGRATIONS newer than the connection's current version, in order."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    current = schema_version(conn)
    for i, migration in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript(migration)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
    conn.commit()
