"""SQLite-backed CRUD layer for memories: the source of truth for content,
scope, status, confidence, and provenance, with every mutation appended to an
audit log. Chroma (vector_index.py) only ever stores embeddings keyed by the
same memory id — this module is where memories actually live.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from yaadein.schema import connect, migrate
from yaadein.types import Episode, Memory

_COLUMNS = [
    "id", "content", "category", "scope_type", "scope_key", "status",
    "confidence", "source_harness", "source_session", "evidence",
    "created_at", "last_retrieved", "times_retrieved", "times_used",
    "superseded_by", "conflict_with", "episode_id",
]

_EPISODE_COLUMNS = [
    "id", "session_id", "source_harness", "scope_type", "scope_key",
    "summary", "excerpt", "transcript_path", "transcript_format",
    "turn_start", "turn_end", "created_at",
]


def _now() -> str:
    """Current UTC timestamp in ISO 8601 format, used for all stored timestamps."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Convert a raw sqlite3.Row from the memories table into a Memory dataclass."""
    return Memory(**{col: row[col] for col in _COLUMNS})


def _row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a raw sqlite3.Row from the episodes table into an Episode dataclass."""
    return Episode(**{col: row[col] for col in _EPISODE_COLUMNS})


class MemoryStore:
    """Owns the SQLite connection and is the sole writer/reader of the
    memories and audit_log tables. Every mutating method records an audit entry."""

    def __init__(self, db_path: Path):
        self._conn = connect(db_path)
        migrate(self._conn)

    def add(self, memory: Memory) -> Memory:
        """Insert a new memory, assigning an id and created_at if not already set."""
        if not memory.id:
            memory.id = f"mem_{uuid.uuid4().hex[:12]}"
        if not memory.created_at:
            memory.created_at = _now()
        values = [getattr(memory, col) for col in _COLUMNS]
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT INTO memories ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self._audit("add", memory.id, memory.content)
        self._conn.commit()
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch one memory by id, or None if it doesn't exist."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def list(
        self,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Memory]:
        """List memories matching any combination of scope_type, scope_key, and
        status filters (all optional; omitted filters are not applied)."""
        clauses, params = [], []
        for col, val in (
            ("scope_type", scope_type), ("scope_key", scope_key), ("status", status)
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at", params
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def record_retrieval(self, memory_ids: List[str]) -> None:
        """Bump times_retrieved and last_retrieved for each memory, e.g. after a recall/briefing."""
        ts = _now()
        for memory_id in memory_ids:
            self._conn.execute(
                "UPDATE memories SET times_retrieved = times_retrieved + 1, "
                "last_retrieved = ? WHERE id = ?",
                (ts, memory_id),
            )
            self._audit("retrieve", memory_id, None)
        self._conn.commit()

    def set_status(self, memory_id: str, status: str) -> None:
        """Transition a memory's status (e.g. proposed -> confirmed -> archived)."""
        self._conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?", (status, memory_id)
        )
        self._audit(f"status:{status}", memory_id, None)
        self._conn.commit()

    def reinforce(self, memory_id: str, source_session: Optional[str] = None) -> None:
        """Nudge confidence up (capped at 1.0) when a near-duplicate fact
        reappears, instead of writing a second memory."""
        self._conn.execute(
            "UPDATE memories SET confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
            (memory_id,),
        )
        self._audit("reinforce", memory_id, source_session)
        self._conn.commit()

    def add_episode(self, episode: Episode) -> Episode:
        """Persist a write-once episode; honors preset id (extractor pre-stamps facts)."""
        if not episode.id:
            episode.id = f"ep_{uuid.uuid4().hex[:12]}"
        if not episode.created_at:
            episode.created_at = _now()
        values = [getattr(episode, col) for col in _EPISODE_COLUMNS]
        placeholders = ", ".join("?" for _ in _EPISODE_COLUMNS)
        self._conn.execute(
            f"INSERT INTO episodes ({', '.join(_EPISODE_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self._audit("add_episode", episode.id, episode.session_id)
        self._conn.commit()
        return episode

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Fetch one episode by id, or None if it doesn't exist."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return _row_to_episode(row) if row else None

    def list_episodes(
        self,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Episode]:
        """List episodes matching any combination of scope_type and scope_key filters
        (all optional; omitted filters are not applied). Newest first by created_at."""
        clauses, params = [], []
        for col, val in (("scope_type", scope_type), ("scope_key", scope_key)):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        tail = f" LIMIT {int(limit)}" if limit is not None else ""
        rows = self._conn.execute(
            f"SELECT * FROM episodes{where} ORDER BY created_at DESC{tail}", params
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def fact_ids_for_episode(self, episode_id: str) -> List[str]:
        """Return ids of all memories stamped with this episode, oldest first."""
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE episode_id = ? ORDER BY created_at",
            (episode_id,),
        ).fetchall()
        return [r["id"] for r in rows]

    def delete_episode(self, episode_id: str) -> None:
        """Rollback helper for a failed index write — episodes are otherwise write-once."""
        self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._audit("rollback_episode", episode_id, None)
        self._conn.commit()

    def forget(self, memory_id: str) -> bool:
        """Permanently delete a memory. Returns False if the id didn't exist."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        if cursor.rowcount == 0:
            return False
        self._audit("forget", memory_id, None)
        self._conn.commit()
        return True

    def audit_entries(self) -> List[sqlite3.Row]:
        """Return the full audit log, oldest first."""
        return self._conn.execute(
            "SELECT * FROM audit_log ORDER BY id"
        ).fetchall()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _audit(self, action: str, memory_id: Optional[str], detail: Optional[str]) -> None:
        """Append one row to audit_log; called by every mutating method above."""
        self._conn.execute(
            "INSERT INTO audit_log (ts, action, memory_id, detail) VALUES (?, ?, ?, ?)",
            (_now(), action, memory_id, detail),
        )
