import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from yaadein.schema import connect, migrate
from yaadein.types import Memory

_COLUMNS = [
    "id", "content", "category", "scope_type", "scope_key", "status",
    "confidence", "source_harness", "source_session", "evidence",
    "created_at", "last_retrieved", "times_retrieved", "times_used",
    "superseded_by", "conflict_with",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(**{col: row[col] for col in _COLUMNS})


class MemoryStore:
    def __init__(self, db_path: Path):
        self._conn = connect(db_path)
        migrate(self._conn)

    def add(self, memory: Memory) -> Memory:
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
        self._conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?", (status, memory_id)
        )
        self._audit(f"status:{status}", memory_id, None)
        self._conn.commit()

    def reinforce(self, memory_id: str, source_session: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE memories SET confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
            (memory_id,),
        )
        self._audit("reinforce", memory_id, source_session)
        self._conn.commit()

    def forget(self, memory_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        if cursor.rowcount == 0:
            return False
        self._audit("forget", memory_id, None)
        self._conn.commit()
        return True

    def audit_entries(self) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM audit_log ORDER BY id"
        ).fetchall()

    def close(self) -> None:
        self._conn.close()

    def _audit(self, action: str, memory_id: Optional[str], detail: Optional[str]) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (ts, action, memory_id, detail) VALUES (?, ?, ?, ?)",
            (_now(), action, memory_id, detail),
        )
