# Agent Memory Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-agent memory layer to Recall: SQLite-backed memory store with scopes and audit log, hybrid recall over memories, and four MCP tools (`remember`, `recall_memory`, `forget_memory`, `memory_briefing`) so any MCP agent shares one brain.

**Architecture:** SQLite is the source of truth (new `memory/` package); a dedicated Chroma collection holds embeddings for semantic search only, linked by memory id. Tool logic lives in plain sync functions (`memory/mcp_tools.py`) that `mcp_server.py` dispatches to, so everything is testable without the MCP transport. Explicit writes land as `confirmed`; the auto-extractor (Plan 2) will land `proposed` rows in the same schema.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib), chromadb (already a dependency), Ollama `nomic-embed-text` via existing llama_index wiring, pytest.

## Global Constraints

- Fully local: no network calls except Ollama at `OLLAMA_BASE_URL` (`http://localhost:11434`).
- No new pip dependencies — stdlib `sqlite3` + existing `chromadb`/`llama_index`.
- All new config constants go in `config.py`; no magic constants in modules.
- Category values: exactly `preference | decision | fact | gotcha`.
- Scope types: exactly `user | project | shared`; user scope_key is `"*"`.
- Status values: exactly `proposed | confirmed | archived`.
- Nothing hard-deleted except `forget`; every mutation writes to `audit_log`.
- Tests never require Ollama running — embeddings are injected via the `Embedder` protocol; tests use `FakeEmbedder`.
- Follow existing test style: plain pytest functions, `tmp_path` fixtures, one behavior per test.

---

### Task 1: Config constants, schema, and migrations

**Files:**
- Modify: `config.py` (append after line 15, `HYBRID_ALPHA = 0.5`)
- Create: `memory/__init__.py` (empty)
- Create: `memory/schema.py`
- Test: `tests/test_memory_schema.py`

**Interfaces:**
- Consumes: `config.BASE_DIR` (existing).
- Produces: `memory.schema.connect(db_path: Path) -> sqlite3.Connection` (creates parent dirs, enables `sqlite3.Row`), `memory.schema.migrate(conn) -> None` (idempotent), `memory.schema.schema_version(conn) -> int`. Tables: `memories`, `audit_log`, `schema_version`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_schema.py
import sqlite3
from memory.schema import connect, migrate, schema_version


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
    assert schema_version(conn) == 1
    conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "memories.db")
    migrate(conn)
    migrate(conn)
    assert schema_version(conn) == 1
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory'`

- [ ] **Step 3: Write minimal implementation**

Append to `config.py`:

```python
# ── Memory layer ──────────────────────────────────────────
MEMORY_DIR = BASE_DIR / "memory_store"
MEMORY_DB_PATH = MEMORY_DIR / "memories.db"
MEMORY_CHROMA_DIR = MEMORY_DIR / "chroma_db"
MEMORY_COLLECTION = "recall_memories"
MEMORY_TOP_K = 5
MEMORY_KEYWORD_BONUS = 0.1
MEMORY_BRIEFING_LIMITS = {"facts": 10, "decisions": 5, "gotchas": 5}
```

Create `memory/__init__.py` (empty file).

Create `memory/schema.py`:

```python
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
]


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return row["version"] if row else 0


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    current = schema_version(conn)
    for i, migration in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript(migration)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_schema.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add config.py memory/__init__.py memory/schema.py tests/test_memory_schema.py
git commit -m "feat(memory): SQLite schema, migrations, and config for memory layer"
```

---

### Task 2: Memory dataclass and MemoryStore

**Files:**
- Create: `memory/types.py`
- Create: `memory/store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `memory.schema.connect`, `memory.schema.migrate` (Task 1).
- Produces:
  - `memory.types.Memory` — dataclass with fields matching the `memories` table exactly (`id: str`, `content: str`, `category: str`, `scope_type: str`, `scope_key: str`, `status: str = "proposed"`, `confidence: float = 1.0`, `source_harness: Optional[str] = None`, `source_session: Optional[str] = None`, `evidence: Optional[str] = None`, `created_at: str = ""`, `last_retrieved: Optional[str] = None`, `times_retrieved: int = 0`, `times_used: int = 0`, `superseded_by: Optional[str] = None`, `conflict_with: Optional[str] = None`).
  - `memory.store.MemoryStore(db_path: Path)` with methods:
    - `add(memory: Memory) -> Memory` (assigns uuid id and created_at if empty; audits `"add"`)
    - `get(memory_id: str) -> Optional[Memory]`
    - `list(scope_type: Optional[str] = None, scope_key: Optional[str] = None, status: Optional[str] = None) -> List[Memory]`
    - `record_retrieval(memory_ids: List[str]) -> None` (bumps `times_retrieved`, sets `last_retrieved`; audits `"retrieve"`)
    - `set_status(memory_id: str, status: str) -> None` (audits `"status:<new>"`)
    - `forget(memory_id: str) -> bool` (deletes row; audits `"forget"`; returns False if id unknown)
    - `audit_entries() -> List[sqlite3.Row]` (ordered by id)
    - `close() -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_store.py
from memory.store import MemoryStore
from memory.types import Memory


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.store'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/types.py`:

```python
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Memory:
    id: str
    content: str
    category: str
    scope_type: str
    scope_key: str
    status: str = "proposed"
    confidence: float = 1.0
    source_harness: Optional[str] = None
    source_session: Optional[str] = None
    evidence: Optional[str] = None
    created_at: str = ""
    last_retrieved: Optional[str] = None
    times_retrieved: int = 0
    times_used: int = 0
    superseded_by: Optional[str] = None
    conflict_with: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
```

Create `memory/store.py`:

```python
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from memory.schema import connect, migrate
from memory.types import Memory

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_store.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full suite to check nothing broke**

Run: `pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add memory/types.py memory/store.py tests/test_memory_store.py
git commit -m "feat(memory): Memory dataclass and MemoryStore with audit log"
```

---

### Task 3: Scope resolution

**Files:**
- Create: `memory/scopes.py`
- Test: `tests/test_memory_scopes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure module; shells out to `git`).
- Produces: `memory.scopes.resolve_project_key(path: str) -> str` — normalized git remote URL if available, else git repo root path, else the absolute path itself. `memory.scopes.USER_SCOPE_KEY = "*"`. Normalization: strip trailing `.git`, strip trailing `/`, lowercase host part is NOT attempted (keep simple: just strip `.git` and trailing slash).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_scopes.py
import subprocess
from memory.scopes import resolve_project_key, USER_SCOPE_KEY


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_user_scope_key_is_star():
    assert USER_SCOPE_KEY == "*"


def test_non_git_dir_resolves_to_absolute_path(tmp_path):
    assert resolve_project_key(str(tmp_path)) == str(tmp_path.resolve())


def test_git_repo_without_remote_resolves_to_repo_root(tmp_path):
    _git("init", cwd=tmp_path)
    subdir = tmp_path / "src"
    subdir.mkdir()
    assert resolve_project_key(str(subdir)) == str(tmp_path.resolve())


def test_git_repo_with_remote_resolves_to_normalized_url(tmp_path):
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/jai/recall.git", cwd=tmp_path)
    assert resolve_project_key(str(tmp_path)) == "https://github.com/jai/recall"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_scopes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.scopes'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/scopes.py`:

```python
import subprocess
from pathlib import Path

USER_SCOPE_KEY = "*"


def _git_output(args, cwd: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _normalize_remote(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def resolve_project_key(path: str) -> str:
    resolved = str(Path(path).resolve())
    remote = _git_output(["remote", "get-url", "origin"], cwd=resolved)
    if remote:
        return _normalize_remote(remote)
    root = _git_output(["rev-parse", "--show-toplevel"], cwd=resolved)
    if root:
        return str(Path(root).resolve())
    return resolved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_scopes.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add memory/scopes.py tests/test_memory_scopes.py
git commit -m "feat(memory): project scope resolution via git remote/root"
```

---

### Task 4: Embedder protocol and memory vector index

**Files:**
- Create: `memory/vector_index.py`
- Test: `tests/test_memory_vector_index.py`

**Interfaces:**
- Consumes: `utils.chroma_client.make_chroma_client` (existing; signature `make_chroma_client(persist_dir: str)` returning a chromadb client).
- Produces:
  - `memory.vector_index.Embedder` — Protocol with `embed(text: str) -> List[float]`.
  - `memory.vector_index.OllamaEmbedder` — implements `Embedder` using `llama_index.embeddings.ollama.OllamaEmbedding` with `config.EMBED_MODEL` / `config.OLLAMA_BASE_URL`.
  - `memory.vector_index.MemoryVectorIndex(chroma_dir: Path, embedder: Embedder, collection_name: str)` with:
    - `add(memory_id: str, content: str) -> None`
    - `query(text: str, top_k: int) -> List[tuple[str, float]]` (id, similarity in [0,1], cosine space, sorted desc)
    - `delete(memory_id: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_vector_index.py
import math
from memory.vector_index import MemoryVectorIndex


class FakeEmbedder:
    """Deterministic 4-dim embeddings keyed on which words appear."""

    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_index(tmp_path):
    return MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_memories",
    )


def test_query_ranks_semantically_closest_first(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    index.add("m2", "Deploys go through the blue pipeline")
    results = index.query("what testing framework? pytest?", top_k=2)
    assert results[0][0] == "m1"
    assert results[0][1] > results[1][1]


def test_scores_are_similarities_between_zero_and_one(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    results = index.query("pytest testing", top_k=1)
    assert 0.0 <= results[0][1] <= 1.0


def test_delete_removes_from_results(tmp_path):
    index = make_index(tmp_path)
    index.add("m1", "User prefers pytest for all testing")
    index.delete("m1")
    assert index.query("pytest", top_k=1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_vector_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.vector_index'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/vector_index.py`:

```python
from pathlib import Path
from typing import List, Protocol, Tuple

from utils.chroma_client import make_chroma_client


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class OllamaEmbedder:
    def __init__(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from config import EMBED_MODEL, OLLAMA_BASE_URL

        self._model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    def embed(self, text: str) -> List[float]:
        return self._model.get_text_embedding(text)


class MemoryVectorIndex:
    def __init__(self, chroma_dir: Path, embedder: Embedder, collection_name: str):
        self._embedder = embedder
        client = make_chroma_client(str(chroma_dir))
        self._collection = client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, memory_id: str, content: str) -> None:
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[self._embedder.embed(content)],
            documents=[content],
        )

    def query(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed(text)],
            n_results=min(top_k, count),
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # cosine distance -> similarity, clamped to [0, 1]
        return [
            (memory_id, max(0.0, min(1.0, 1.0 - dist)))
            for memory_id, dist in zip(ids, distances)
        ]

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_vector_index.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add memory/vector_index.py tests/test_memory_vector_index.py
git commit -m "feat(memory): Embedder protocol and Chroma-backed memory vector index"
```

---

### Task 5: MemoryService — remember, recall, forget

**Files:**
- Create: `memory/service.py`
- Test: `tests/test_memory_service.py`

**Interfaces:**
- Consumes: `MemoryStore` (Task 2), `MemoryVectorIndex`/`Embedder` (Task 4), `resolve_project_key`/`USER_SCOPE_KEY` (Task 3), `config.MEMORY_TOP_K`, `config.MEMORY_KEYWORD_BONUS`.
- Produces: `memory.service.MemoryService(store: MemoryStore, vector_index: MemoryVectorIndex)` with:
  - `remember(content: str, category: str = "fact", scope_type: str = "user", scope_key: str = "*", source_harness: Optional[str] = None, source_session: Optional[str] = None) -> Memory` — status `confirmed`, confidence 1.0.
  - `recall(query: str, project_key: Optional[str] = None, top_k: Optional[int] = None) -> List[dict]` — each dict is `Memory.to_dict()` plus `"score": float`; searches user scope always, plus project scope when `project_key` given; excludes `archived` and superseded rows; keyword bonus of `MEMORY_KEYWORD_BONUS` per query term (len > 3) found in content, capped at 0.3; records retrieval on returned ids.
  - `forget(memory_id: str) -> bool` — removes from store and vector index.
  - Also: `memory.service.get_memory_service() -> MemoryService` — module-level singleton factory wiring real config paths + `OllamaEmbedder` (used by the MCP server; not unit-tested since it needs Ollama).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_service.py
import math
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=FakeEmbedder(),
        collection_name="test_memories",
    )
    return MemoryService(store=store, vector_index=index), store


def test_remember_lands_confirmed(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest", category="preference")
    assert store.get(saved.id).status == "confirmed"


def test_recall_returns_scored_dicts_ranked(tmp_path):
    service, _ = make_service(tmp_path)
    service.remember("User prefers pytest for testing", category="preference")
    service.remember("Deploys go through the blue pipeline", category="fact")
    results = service.recall("which pytest testing setup?")
    assert results[0]["content"] == "User prefers pytest for testing"
    assert results[0]["score"] > results[1]["score"]


def test_recall_excludes_archived(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    store.set_status(saved.id, "archived")
    assert service.recall("pytest") == []


def test_recall_scopes_project_memories(tmp_path):
    service, _ = make_service(tmp_path)
    service.remember(
        "Auth module is fragile", category="gotcha",
        scope_type="project", scope_key="repo-a",
    )
    with_project = service.recall("auth fragile?", project_key="repo-a")
    without_project = service.recall("auth fragile?", project_key="repo-b")
    assert len(with_project) == 1
    assert without_project == []


def test_recall_records_retrieval(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    service.recall("pytest")
    assert store.get(saved.id).times_retrieved == 1


def test_forget_removes_everywhere(tmp_path):
    service, _ = make_service(tmp_path)
    saved = service.remember("User prefers pytest for testing")
    assert service.forget(saved.id) is True
    assert service.recall("pytest") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.service'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/service.py`:

```python
from typing import List, Optional

from config import (
    MEMORY_CHROMA_DIR, MEMORY_COLLECTION, MEMORY_DB_PATH,
    MEMORY_KEYWORD_BONUS, MEMORY_TOP_K,
)
from memory.scopes import USER_SCOPE_KEY
from memory.store import MemoryStore
from memory.types import Memory
from memory.vector_index import MemoryVectorIndex

_KEYWORD_BONUS_CAP = 0.3


class MemoryService:
    def __init__(self, store: MemoryStore, vector_index: MemoryVectorIndex):
        self._store = store
        self._index = vector_index

    def remember(
        self,
        content: str,
        category: str = "fact",
        scope_type: str = "user",
        scope_key: str = USER_SCOPE_KEY,
        source_harness: Optional[str] = None,
        source_session: Optional[str] = None,
    ) -> Memory:
        memory = Memory(
            id="", content=content, category=category,
            scope_type=scope_type, scope_key=scope_key,
            status="confirmed", confidence=1.0,
            source_harness=source_harness, source_session=source_session,
        )
        saved = self._store.add(memory)
        self._index.add(saved.id, saved.content)
        return saved

    def recall(
        self,
        query: str,
        project_key: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        top_k = top_k or MEMORY_TOP_K
        # over-fetch so scope filtering still leaves top_k candidates
        hits = self._index.query(query, top_k=top_k * 4)
        terms = [t for t in query.lower().split() if len(t) > 3]

        scored = []
        for memory_id, similarity in hits:
            memory = self._store.get(memory_id)
            if memory is None:
                continue
            if memory.status == "archived" or memory.superseded_by:
                continue
            if not self._in_scope(memory, project_key):
                continue
            bonus = min(
                _KEYWORD_BONUS_CAP,
                MEMORY_KEYWORD_BONUS
                * sum(1 for t in terms if t in memory.content.lower()),
            )
            scored.append((memory, similarity + bonus))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:top_k]
        self._store.record_retrieval([m.id for m, _ in top])
        return [{**m.to_dict(), "score": round(score, 4)} for m, score in top]

    def forget(self, memory_id: str) -> bool:
        removed = self._store.forget(memory_id)
        if removed:
            self._index.delete(memory_id)
        return removed

    @staticmethod
    def _in_scope(memory: Memory, project_key: Optional[str]) -> bool:
        if memory.scope_type == "user":
            return True
        if memory.scope_type == "project":
            return project_key is not None and memory.scope_key == project_key
        return False  # shared scope arrives with the extractor/live workspaces


_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    global _service
    if _service is None:
        from memory.vector_index import OllamaEmbedder

        _service = MemoryService(
            store=MemoryStore(MEMORY_DB_PATH),
            vector_index=MemoryVectorIndex(
                chroma_dir=MEMORY_CHROMA_DIR,
                embedder=OllamaEmbedder(),
                collection_name=MEMORY_COLLECTION,
            ),
        )
    return _service
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_service.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add memory/service.py tests/test_memory_service.py
git commit -m "feat(memory): MemoryService with remember/recall/forget and scoped hybrid ranking"
```

---

### Task 6: memory_briefing

**Files:**
- Modify: `memory/service.py` (add `briefing` method to `MemoryService`)
- Test: `tests/test_memory_briefing.py`

**Interfaces:**
- Consumes: `MemoryService`, `MemoryStore.list` (Task 2), `config.MEMORY_BRIEFING_LIMITS`.
- Produces: `MemoryService.briefing(project_key: Optional[str] = None) -> dict` with keys:
  - `"facts"`: top confirmed `preference`/`fact` memories (user scope + project scope), sorted by `times_retrieved` desc, limit `MEMORY_BRIEFING_LIMITS["facts"]`
  - `"decisions"`: confirmed `decision` memories, newest first, limit `["decisions"]`
  - `"gotchas"`: confirmed + proposed `gotcha` memories, newest first, limit `["gotchas"]`; proposed items get `"unconfirmed": True` in their dict
  - `"conflicts"`: memories with non-null `conflict_with`
  - Every returned memory dict is `Memory.to_dict()`; briefing records retrieval on all returned ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_briefing.py
import math
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.types import Memory
from memory.vector_index import MemoryVectorIndex


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_briefing.py -v`
Expected: FAIL with `AttributeError: 'MemoryService' object has no attribute 'briefing'`

- [ ] **Step 3: Write minimal implementation**

Add to `memory/service.py` — import at top:

```python
from config import (
    MEMORY_BRIEFING_LIMITS, MEMORY_CHROMA_DIR, MEMORY_COLLECTION,
    MEMORY_DB_PATH, MEMORY_KEYWORD_BONUS, MEMORY_TOP_K,
)
```

Add method to `MemoryService` (after `forget`):

```python
    def briefing(self, project_key: Optional[str] = None) -> dict:
        candidates = [
            m for m in self._store.list()
            if m.status != "archived"
            and not m.superseded_by
            and self._in_scope(m, project_key)
        ]

        def to_dict(memory: Memory) -> dict:
            d = memory.to_dict()
            if memory.status == "proposed":
                d["unconfirmed"] = True
            return d

        confirmed = [m for m in candidates if m.status == "confirmed"]
        facts = sorted(
            (m for m in confirmed if m.category in ("preference", "fact")),
            key=lambda m: m.times_retrieved, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["facts"]]
        decisions = sorted(
            (m for m in confirmed if m.category == "decision"),
            key=lambda m: m.created_at, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["decisions"]]
        gotchas = sorted(
            (m for m in candidates if m.category == "gotcha"),
            key=lambda m: m.created_at, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["gotchas"]]
        conflicts = [m for m in candidates if m.conflict_with]

        returned = facts + decisions + gotchas + conflicts
        self._store.record_retrieval([m.id for m in returned])
        return {
            "facts": [to_dict(m) for m in facts],
            "decisions": [to_dict(m) for m in decisions],
            "gotchas": [to_dict(m) for m in gotchas],
            "conflicts": [to_dict(m) for m in conflicts],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_briefing.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add memory/service.py tests/test_memory_briefing.py
git commit -m "feat(memory): session-start memory briefing"
```

---

### Task 7: MCP tools and server wiring

**Files:**
- Create: `memory/mcp_tools.py`
- Modify: `mcp_server.py` (extend `handle_list_tools` at line 202 and `handle_call_tool` at line 235)
- Test: `tests/test_memory_mcp_tools.py`

**Interfaces:**
- Consumes: `MemoryService` (Tasks 5–6), `resolve_project_key` (Task 3), `mcp.types` (existing dependency).
- Produces:
  - `memory.mcp_tools.memory_tool_definitions() -> list[mcp.types.Tool]` — four tools: `remember`, `recall_memory`, `forget_memory`, `memory_briefing`.
  - `memory.mcp_tools.handle_memory_tool(name: str, arguments: dict, service: MemoryService) -> Optional[str]` — returns a JSON string result, or `None` if `name` is not a memory tool (so the server falls through to document tools).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_mcp_tools.py
import json
import math
from memory.mcp_tools import memory_tool_definitions, handle_memory_tool
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_service(tmp_path):
    return MemoryService(
        store=MemoryStore(tmp_path / "memories.db"),
        vector_index=MemoryVectorIndex(
            chroma_dir=tmp_path / "chroma",
            embedder=FakeEmbedder(),
            collection_name="test_memories",
        ),
    )


def test_tool_definitions_expose_four_tools():
    names = {t.name for t in memory_tool_definitions()}
    assert names == {"remember", "recall_memory", "forget_memory", "memory_briefing"}


def test_non_memory_tool_returns_none(tmp_path):
    assert handle_memory_tool("query_rag", {}, make_service(tmp_path)) is None


def test_remember_then_recall_roundtrip(tmp_path):
    service = make_service(tmp_path)
    remembered = json.loads(handle_memory_tool(
        "remember",
        {"content": "User prefers pytest", "category": "preference"},
        service,
    ))
    assert remembered["status"] == "confirmed"

    recalled = json.loads(handle_memory_tool(
        "recall_memory", {"query": "pytest testing"}, service
    ))
    assert recalled[0]["content"] == "User prefers pytest"


def test_forget_memory_reports_result(tmp_path):
    service = make_service(tmp_path)
    remembered = json.loads(handle_memory_tool(
        "remember", {"content": "temp fact"}, service
    ))
    result = json.loads(handle_memory_tool(
        "forget_memory", {"memory_id": remembered["id"]}, service
    ))
    assert result == {"forgotten": True}


def test_memory_briefing_returns_sections(tmp_path):
    service = make_service(tmp_path)
    handle_memory_tool(
        "remember", {"content": "User prefers pytest", "category": "preference"},
        service,
    )
    briefing = json.loads(handle_memory_tool("memory_briefing", {}, service))
    assert set(briefing) == {"facts", "decisions", "gotchas", "conflicts"}
    assert briefing["facts"][0]["content"] == "User prefers pytest"


def test_missing_required_argument_returns_error(tmp_path):
    result = json.loads(handle_memory_tool("remember", {}, make_service(tmp_path)))
    assert "error" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_mcp_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.mcp_tools'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/mcp_tools.py`:

```python
import json
from typing import Optional

from mcp import types

from memory.scopes import resolve_project_key
from memory.service import MemoryService

_MEMORY_TOOLS = {"remember", "recall_memory", "forget_memory", "memory_briefing"}


def memory_tool_definitions() -> list:
    return [
        types.Tool(
            name="recall_memory",
            description=(
                "Search the user's persistent cross-agent memory for preferences, "
                "past decisions, project conventions, and gotchas. Call this BEFORE "
                "assuming what the user prefers or how this project works. "
                "Pass project_path to include project-scoped memories."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path of the current project (optional).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="remember",
            description=(
                "Save a durable fact to the user's persistent memory, shared across "
                "all AI agents. Use for preferences, decisions with reasons, and "
                "project gotchas the user states or confirms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "One distilled fact."},
                    "category": {
                        "type": "string",
                        "enum": ["preference", "decision", "fact", "gotcha"],
                        "description": "Kind of fact (default: fact).",
                    },
                    "project_path": {
                        "type": "string",
                        "description": (
                            "If this fact is specific to a project, its absolute path; "
                            "omit for user-wide facts."
                        ),
                    },
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="forget_memory",
            description="Permanently delete a memory by id (from recall_memory results).",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory id to delete."},
                },
                "required": ["memory_id"],
            },
        ),
        types.Tool(
            name="memory_briefing",
            description=(
                "Get a session-start digest of what is known: top user preferences "
                "and facts, recent decisions, active gotchas, and unresolved "
                "conflicts. Call once at the start of a session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path of the current project (optional).",
                    },
                },
                "required": [],
            },
        ),
    ]


def handle_memory_tool(
    name: str, arguments: dict, service: MemoryService
) -> Optional[str]:
    if name not in _MEMORY_TOOLS:
        return None
    try:
        return json.dumps(_dispatch(name, arguments, service))
    except KeyError as e:
        return json.dumps({"error": f"Missing required argument: {e.args[0]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _project_key(arguments: dict) -> Optional[str]:
    path = arguments.get("project_path")
    return resolve_project_key(path) if path else None


def _dispatch(name: str, arguments: dict, service: MemoryService) -> object:
    if name == "remember":
        content = arguments["content"]
        project_key = _project_key(arguments)
        memory = service.remember(
            content=content,
            category=arguments.get("category", "fact"),
            scope_type="project" if project_key else "user",
            scope_key=project_key or "*",
        )
        return memory.to_dict()

    if name == "recall_memory":
        return service.recall(arguments["query"], project_key=_project_key(arguments))

    if name == "forget_memory":
        return {"forgotten": service.forget(arguments["memory_id"])}

    # memory_briefing
    return service.briefing(project_key=_project_key(arguments))
```

Modify `mcp_server.py` — add imports after line 46 (`from utils.ollama_check import check_ollama_running`):

```python
from memory.mcp_tools import memory_tool_definitions, handle_memory_tool
from memory.service import get_memory_service
```

In `handle_list_tools` (line 202), change the return to append memory tools:

```python
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_indexes",
            ...existing definition unchanged...
        ),
        types.Tool(
            name="query_rag",
            ...existing definition unchanged...
        ),
        *memory_tool_definitions(),
    ]
```

In `handle_call_tool` (line 235), add memory dispatch as the FIRST check, before `if name == "list_indexes":`:

```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    memory_result = handle_memory_tool(name, arguments or {}, get_memory_service())
    if memory_result is not None:
        return [types.TextContent(type="text", text=memory_result)]

    if name == "list_indexes":
        ...existing code unchanged...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_mcp_tools.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass (existing `test_ui_endpoints.py` confirms the server still imports and serves)

- [ ] **Step 6: Commit**

```bash
git add memory/mcp_tools.py mcp_server.py tests/test_memory_mcp_tools.py
git commit -m "feat(memory): expose remember/recall/forget/briefing as MCP tools"
```

---

### Task 8: README documentation

**Files:**
- Modify: `README.md` (add a "Cross-Agent Memory" section after the "How It Works" section)

**Interfaces:**
- Consumes: tool names and behavior from Task 7.
- Produces: user-facing docs; no code.

- [ ] **Step 1: Add the section**

Add to `README.md` after the "How It Works" section:

```markdown
## Cross-Agent Memory

Recall now gives all your MCP agents one shared, persistent memory. Facts you
save from Claude Code are known to Cursor and any other MCP client — locally,
with full provenance.

**Tools (available to any connected MCP agent):**

| Tool | What it does |
|---|---|
| `remember` | Save a durable fact (preference, decision, fact, gotcha) |
| `recall_memory` | Search memories, ranked; pass `project_path` for project scope |
| `memory_briefing` | Session-start digest: top facts, recent decisions, gotchas |
| `forget_memory` | Permanently delete a memory by id |

**Scopes:** memories are either user-wide (`"*"`) or bound to a project
(keyed by git remote URL, falling back to repo root path).

**Storage:** SQLite (`memory_store/memories.db`) is the source of truth;
embeddings live in a dedicated Chroma collection. Every mutation is recorded
in an audit log. Nothing leaves your machine.

Auto-extraction of memories from session transcripts is the next phase — see
`docs/specs/2026-07-03-agent-memory-design.md`.
```

- [ ] **Step 2: Add memory_store to .gitignore**

Append to `.gitignore` under the "User data" section:

```
# ── Memory layer data (user data, never commit) ───────────
memory_store/
```

- [ ] **Step 3: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: document cross-agent memory tools"
```

---

## Out of scope for this plan (Plan 2: extractor + lifecycle + inspector)

Per the spec's build order, steps 4–6 (extractor pipeline + SessionEnd hook +
eval set; lifecycle engine; inspector CLI) are a separate plan, written after
this core ships. The schema above already carries their fields
(`status='proposed'`, `evidence`, `superseded_by`, `conflict_with`,
`times_used`), so Plan 2 requires no migration of existing data — only new
`MIGRATIONS` entries if columns are added.
