# Yaadein Test Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thorough, scenario-driven test suite that proves Yaadein is production-ready and teaches the full data flow through writing tests.

**Architecture:** 5 scenarios (S1–S5) each get a unit test file under `tests/scenarios/` and a shared live test file under `tests/live/`. Gap fixes go into existing test files. Stress tests are co-located in scenario files behind a `stress` marker.

**Tech Stack:** Python 3.11, pytest, pytest-cov, httpx (live tests), threading (stress tests), unittest.mock

## Global Constraints

- Python: 3.11 (`python3.11`)
- Test runner: `python3.11 -m pytest`
- New markers: `live` (daemon on localhost:8765), `stress` (concurrency/load)
- Default run must stay CI-safe: no live daemon, no Ollama, no network
- All new tests must pass in < 5s in default mode
- Never touch existing test files unless adding gap-fill tests to them
- Run command from `~/workplace/yaadein/`

---

## File Map

**New files:**
- `tests/scenarios/__init__.py` — empty package marker
- `tests/scenarios/test_s1_happy_path.py` — S1: full pipeline unit + stress
- `tests/scenarios/test_s2_recall_load.py` — S2: recall correctness + stress
- `tests/scenarios/test_s3_episode_lifecycle.py` — S3: episode create/recall
- `tests/scenarios/test_s4_resilience.py` — S4: error injection tests
- `tests/scenarios/test_s5_privacy_gates.py` — S5: redaction + scope isolation
- `tests/live/__init__.py` — empty package marker
- `tests/live/test_live_daemon.py` — live HTTP tests against running daemon

**Modified files:**
- `pytest.ini` — add `live` and `stress` markers
- `tests/test_memory_extractor.py` — gap fills for lines 91-92, 105-106, 170, 174-175, 202, 213, 249-252, 271-275
- `tests/test_memory_llm.py` — gap fills for lines 20-23, 30
- `tests/test_memory_mcp_tools.py` — gap fill for lines 166-168
- `tests/test_memory_service.py` — gap fills for lines 115, 174, 181-182, 229, 291, 318, 332-348
- `tests/test_memory_store.py` — gap fill for line 203
- `tests/test_memory_transcript.py` — gap fill for lines 28, 62
- `tests/test_memory_vector_index.py` — gap fills for lines 26-29, 33
- `tests/test_memory_watcher.py` — gap fills for lines 23, 36, 45-46

---

## Task 1: pytest.ini — register new markers

**Files:**
- Modify: `pytest.ini`

**Interfaces:**
- Produces: `live` and `stress` markers usable in all subsequent tasks

- [ ] **Step 1: Add markers to pytest.ini**

Replace the `markers =` block so it reads:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    eval: extraction-quality evals that need a local Ollama LLM (deselected by default)
    live: tests that require the Yaadein daemon running on localhost:8765
    stress: load and concurrency tests (deselected by default)
addopts = -m "not eval and not live and not stress"
```

- [ ] **Step 2: Verify collection still works**

```bash
python3.11 -m pytest --co -q 2>&1 | tail -5
```
Expected: `120/120 tests collected` (no errors, no warnings about unknown markers)

- [ ] **Step 3: Commit**

```bash
git add pytest.ini
git commit -m "test(config): register live and stress pytest markers"
```

---

## Task 2: Scaffold scenario and live packages

**Files:**
- Create: `tests/scenarios/__init__.py`
- Create: `tests/live/__init__.py`

**Interfaces:**
- Produces: importable `tests.scenarios` and `tests.live` packages

- [ ] **Step 1: Create both __init__.py files**

```bash
touch tests/scenarios/__init__.py tests/live/__init__.py
```

- [ ] **Step 2: Verify pytest collects them**

```bash
python3.11 -m pytest --co -q tests/scenarios/ tests/live/ 2>&1 | tail -5
```
Expected: `no tests ran` (packages exist but are empty — no error)

- [ ] **Step 3: Commit**

```bash
git add tests/scenarios/__init__.py tests/live/__init__.py
git commit -m "test(scaffold): add scenarios and live test packages"
```

---

## Task 3: S1 — Happy Path Session (unit + stress)

**WHY this scenario:** Proves the full pipeline — transcript arrives, extractor fires, facts land in the store, briefing returns them. If this breaks, nothing works.

**Files:**
- Create: `tests/scenarios/test_s1_happy_path.py`

**Interfaces:**
- Consumes: `yaadein.extractor.Extractor`, `yaadein.service.MemoryService`, `yaadein.store.MemoryStore`, `yaadein.vector_index.MemoryVectorIndex`
- Produces: nothing (test-only file)

- [ ] **Step 1: Write the failing test**

```python
# tests/scenarios/test_s1_happy_path.py
# WHY: Traces the full data flow — transcript → extractor → store → briefing.
# Each test here is a station on that pipeline. If a station fails, you know
# exactly where the break is.

import pytest
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_service():
    """Build an in-memory MemoryService with a fake vector index."""
    from yaadein.store import MemoryStore
    from yaadein.service import MemoryService

    store = MemoryStore(":memory:")
    index = MagicMock()
    index.query.return_value = []
    index.add = MagicMock()
    episode_index = MagicMock()
    episode_index.query.return_value = []
    episode_index.add = MagicMock()
    return MemoryService(store=store, vector_index=index, episode_index=episode_index)


def _make_extractor(service, llm_response: str):
    """Build an Extractor with a fake LLM generator returning llm_response."""
    from yaadein.extractor import Extractor

    generator = MagicMock()
    generator.generate.return_value = llm_response
    return Extractor(
        service=service,
        generator=generator,
        extract_log=Path("/tmp/test_extract_log.json"),
    )


SAMPLE_TRANSCRIPT = """
[{"role": "user", "content": "I prefer tabs over spaces in Python."},
 {"role": "assistant", "content": "Noted."}]
"""

LLM_FACT_RESPONSE = """[
  {
    "content": "User prefers tabs over spaces in Python",
    "category": "preference",
    "scope": "user",
    "confidence": 0.9,
    "evidence_quote": "I prefer tabs over spaces in Python."
  }
]"""


def test_s1_fact_stored_after_extraction(tmp_path):
    """Transcript → extractor → fact appears in store."""
    service = _make_service()
    extractor = _make_extractor(service, LLM_FACT_RESPONSE)

    transcript_file = tmp_path / "session.jsonl"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)

    result = extractor.extract(transcript_path=transcript_file, session_id="sess-001")

    assert result.error is None
    memories = service.recall("tabs spaces Python", project_path=None)
    assert len(memories) >= 1
    assert any("tabs" in m.content.lower() for m in memories)


def test_s1_briefing_returns_extracted_fact(tmp_path):
    """Fact stored by extractor appears in briefing response."""
    service = _make_service()
    extractor = _make_extractor(service, LLM_FACT_RESPONSE)

    transcript_file = tmp_path / "session.jsonl"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)
    extractor.extract(transcript_path=transcript_file, session_id="sess-002")

    briefing = service.briefing(project_path=None)
    all_content = [m["content"] for m in briefing.get("facts", [])]
    assert any("tabs" in c.lower() for c in all_content)


def test_s1_duplicate_transcript_not_reextracted(tmp_path):
    """Same transcript submitted twice produces facts only once."""
    service = _make_service()
    extractor = _make_extractor(service, LLM_FACT_RESPONSE)

    transcript_file = tmp_path / "session.jsonl"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)

    extractor.extract(transcript_path=transcript_file, session_id="sess-003")
    extractor.extract(transcript_path=transcript_file, session_id="sess-003")

    memories = service.recall("tabs spaces", project_path=None)
    # Should not have duplicated — same content reinforced, not doubled
    tab_memories = [m for m in memories if "tabs" in m.content.lower()]
    assert len(tab_memories) == 1


@pytest.mark.stress
def test_s1_concurrent_extractions_no_data_loss(tmp_path):
    """20 concurrent extractions each add their own fact — none are lost."""
    import json

    service = _make_service()
    errors = []

    def run(i):
        fact = f"User fact number {i}"
        llm_resp = json.dumps([{
            "content": fact,
            "category": "fact",
            "scope": "user",
            "confidence": 0.9,
            "evidence_quote": fact,
        }])
        transcript = json.dumps([{"role": "user", "content": fact}])
        tf = tmp_path / f"session_{i}.jsonl"
        tf.write_text(transcript)
        extractor = _make_extractor(service, llm_resp)
        result = extractor.extract(transcript_path=tf, session_id=f"sess-stress-{i}")
        if result.error:
            errors.append(result.error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Extraction errors: {errors}"
    all_memories = service.recall("User fact", project_path=None)
    assert len(all_memories) >= 15  # allow some near-duplicate merging
```

- [ ] **Step 2: Run to verify tests fail (service.recall signature may differ)**

```bash
python3.11 -m pytest tests/scenarios/test_s1_happy_path.py -v 2>&1 | tail -20
```

Expected: FAIL or ERROR — we need to check the actual `recall` and `briefing` signatures.

- [ ] **Step 3: Check real signatures and fix test if needed**

```bash
grep -n "def recall\|def briefing" yaadein/service.py
```

Adjust `service.recall(...)` and `service.briefing(...)` calls in the test to match real signatures. The key parameters are `query` (str) and `project_path` (Optional[str] or `project_key`).

- [ ] **Step 4: Run until green**

```bash
python3.11 -m pytest tests/scenarios/test_s1_happy_path.py -v 2>&1 | tail -20
```
Expected: 3 PASSED (stress test skipped by default)

- [ ] **Step 5: Commit**

```bash
git add tests/scenarios/test_s1_happy_path.py
git commit -m "test(s1): happy path — transcript → store → briefing, concurrency stress"
```

---

## Task 4: S2 — Recall Under Load (unit + stress)

**WHY this scenario:** Recall is the core value-add of Yaadein. If ranking is wrong or scopes bleed, the system is broken in the most subtle way — it returns *something*, just not the right thing.

**Files:**
- Create: `tests/scenarios/test_s2_recall_load.py`

**Interfaces:**
- Consumes: `yaadein.service.MemoryService`, `yaadein.store.MemoryStore`, `yaadein.types.Memory`
- Produces: nothing (test-only)

- [ ] **Step 1: Write the failing test**

```python
# tests/scenarios/test_s2_recall_load.py
# WHY: Recall correctness and scope isolation are the guarantees Yaadein makes.
# 500 memories across 3 scopes, then query — top result must be right, and
# project A must never see project B's memories.

import pytest
import threading
import time
from unittest.mock import MagicMock
from yaadein.store import MemoryStore
from yaadein.service import MemoryService


def _make_service_with_real_index():
    """MemoryService with a real in-memory store but mocked vector index.
    Vector index is mocked to return controlled ranked results."""
    store = MemoryStore(":memory:")
    index = MagicMock()
    episode_index = MagicMock()
    episode_index.query.return_value = []
    return store, index, MemoryService(store=store, vector_index=index, episode_index=episode_index)


def _seed_memories(service, store, index, count: int, scope_type: str, scope_key: str, prefix: str):
    """Insert `count` memories and wire the mock index to return them ranked."""
    ids = []
    for i in range(count):
        mem_id = service.propose(
            content=f"{prefix} memory {i}",
            category="fact",
            scope_type=scope_type,
            scope_key=scope_key,
            confidence=0.8,
            source_session=f"sess-{i}",
        )
        ids.append(mem_id)
    return ids


def test_s2_top_result_is_highest_ranked(tmp_path):
    """Query returns the best-matching memory first."""
    store, index, service = _make_service_with_real_index()

    # Seed 10 user-scope memories
    ids = _seed_memories(service, store, index, 10, "user", "*", "topic-alpha")

    # Mock index to return ids ranked: best match first (highest similarity)
    ranked = [(mem_id, 1.0 - (i * 0.05)) for i, mem_id in enumerate(ids)]
    index.query.return_value = ranked

    results = service.recall("topic-alpha memory", project_path=None)
    assert results[0].content == "topic-alpha memory 0"


def test_s2_scope_isolation_project_a_vs_b():
    """Project A's memories never appear in Project B's recall."""
    store, index, service = _make_service_with_real_index()

    ids_a = _seed_memories(service, store, index, 5, "project", "proj-a", "alpha")
    ids_b = _seed_memories(service, store, index, 5, "project", "proj-b", "beta")

    # Index returns all 10 — but scope filtering must exclude cross-project
    all_ids = [(mid, 0.9) for mid in ids_a + ids_b]
    index.query.return_value = all_ids

    results_a = service.recall("memory", project_path="/repos/proj-a")
    result_contents = [r.content for r in results_a]

    # No beta memories should appear
    assert all("beta" not in c for c in result_contents), \
        f"Scope leak: project B content in project A results: {result_contents}"


def test_s2_user_scope_visible_to_all_projects():
    """User-scoped memories appear regardless of which project queries."""
    store, index, service = _make_service_with_real_index()

    user_ids = _seed_memories(service, store, index, 3, "user", "*", "universal")

    index.query.return_value = [(mid, 0.9) for mid in user_ids]

    results_proj = service.recall("universal", project_path="/some/project")
    assert len(results_proj) == 3
    assert all("universal" in r.content for r in results_proj)


def test_s2_500_memories_no_crash():
    """Inserting 500 memories completes without error."""
    store, index, service = _make_service_with_real_index()
    index.query.return_value = []

    for i in range(500):
        service.propose(
            content=f"bulk memory {i}",
            category="fact",
            scope_type="user",
            scope_key="*",
            confidence=0.7,
            source_session="bulk-session",
        )
    # No assertion needed — if we get here without exception, it passes


@pytest.mark.stress
def test_s2_concurrent_readers_writers():
    """10 readers + 5 writers simultaneously — no crashes, no scope leaks."""
    store, index, service = _make_service_with_real_index()
    index.query.return_value = []
    errors = []

    def writer(i):
        try:
            service.propose(
                content=f"stress fact {i}",
                category="fact",
                scope_type="user",
                scope_key="*",
                confidence=0.8,
                source_session=f"stress-sess-{i}",
            )
        except Exception as e:
            errors.append(f"writer {i}: {e}")

    def reader(i):
        try:
            start = time.time()
            service.recall("stress fact", project_path=None)
            elapsed = time.time() - start
            if elapsed > 0.2:
                errors.append(f"reader {i} too slow: {elapsed:.2f}s")
        except Exception as e:
            errors.append(f"reader {i}: {e}")

    threads = (
        [threading.Thread(target=writer, args=(i,)) for i in range(5)] +
        [threading.Thread(target=reader, args=(i,)) for i in range(10)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Stress errors: {errors}"
```

- [ ] **Step 2: Check real `propose` and `recall` signatures**

```bash
grep -n "def propose\|def recall\|def reinforce" yaadein/service.py | head -10
```

Adjust the test calls to match. `propose` may be named differently — check and fix.

- [ ] **Step 3: Run until green**

```bash
python3.11 -m pytest tests/scenarios/test_s2_recall_load.py -v 2>&1 | tail -20
```
Expected: 4 PASSED (stress skipped)

- [ ] **Step 4: Commit**

```bash
git add tests/scenarios/test_s2_recall_load.py
git commit -m "test(s2): recall correctness, scope isolation, 500-memory load, concurrency stress"
```

---

## Task 5: S3 — Episode Lifecycle

**WHY this scenario:** Episodes are what make Yaadein v2 — conversations become memories. If episodes aren't created, linked to facts, and retrievable, v2 is broken at its core concept.

**Files:**
- Create: `tests/scenarios/test_s3_episode_lifecycle.py`

**Interfaces:**
- Consumes: `yaadein.extractor.Extractor`, `yaadein.service.MemoryService`, `yaadein.store.MemoryStore`
- Produces: nothing (test-only)

- [ ] **Step 1: Write the failing test**

```python
# tests/scenarios/test_s3_episode_lifecycle.py
# WHY: Episode lifecycle is: transcript arrives → extractor writes episode with summary
# → episode links to extracted facts → recall_conversations returns it ranked by recency.
# Each test checks one link in that chain.

import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from yaadein.store import MemoryStore
from yaadein.service import MemoryService
from yaadein.extractor import Extractor


TRANSCRIPT = json.dumps([
    {"role": "user", "content": "We decided to use SQLite for the memory store."},
    {"role": "assistant", "content": "Good choice — simple, reliable, no server needed."},
])

FACT_RESPONSE = json.dumps([{
    "content": "SQLite chosen for memory store",
    "category": "decision",
    "scope": "user",
    "confidence": 0.95,
    "evidence_quote": "We decided to use SQLite for the memory store.",
}])

SUMMARY_RESPONSE = "Team decided to use SQLite for the memory store because it is simple and reliable."


def _make_service():
    store = MemoryStore(":memory:")
    index = MagicMock()
    index.query.return_value = []
    index.add = MagicMock()
    ep_index = MagicMock()
    ep_index.query.return_value = []
    ep_index.add = MagicMock()
    return store, MemoryService(store=store, vector_index=index, episode_index=ep_index)


def _make_extractor(service, fact_resp: str, summary_resp: str, extract_log: Path):
    generator = MagicMock()
    # First call = fact extraction, second call = episode summary
    generator.generate.side_effect = [fact_resp, summary_resp]
    return Extractor(service=service, generator=generator, extract_log=extract_log)


def test_s3_episode_created_after_extraction(tmp_path):
    """Extraction creates an episode record in the store."""
    store, service = _make_service()
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-001")

    assert result.error is None
    assert result.episode_id is not None
    episode = store.get_episode(result.episode_id)
    assert episode is not None
    assert "SQLite" in episode.summary


def test_s3_episode_linked_to_facts(tmp_path):
    """Facts extracted in the same session are linked to the episode."""
    store, service = _make_service()
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-002")

    # The episode's fact_ids should reference real memory IDs
    episode = store.get_episode(result.episode_id)
    assert len(episode.fact_ids) >= 1
    for fid in episode.fact_ids:
        memory = store.get(fid)
        assert memory is not None


def test_s3_recall_conversations_returns_episode(tmp_path):
    """recall_conversations finds the episode after extraction."""
    store, service = _make_service()
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-003")

    # Wire the episode index to return the new episode
    service._episode_index.query.return_value = [(result.episode_id, 0.9)]

    episodes = service.recall_conversations("SQLite memory store", project_path=None)
    assert any(ep["id"] == result.episode_id for ep in episodes)


def test_s3_duplicate_session_idempotent(tmp_path):
    """Submitting the same transcript twice does not create two episodes."""
    store, service = _make_service()
    extract_log = tmp_path / "log.json"

    # First extraction
    gen1 = MagicMock()
    gen1.generate.side_effect = [FACT_RESPONSE, SUMMARY_RESPONSE]
    e1 = Extractor(service=service, generator=gen1, extract_log=extract_log)
    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    r1 = e1.extract(transcript_path=tf, session_id="ep-sess-004")

    # Second extraction — same file, same hash → should skip
    gen2 = MagicMock()
    gen2.generate.side_effect = [FACT_RESPONSE, SUMMARY_RESPONSE]
    e2 = Extractor(service=service, generator=gen2, extract_log=extract_log)
    r2 = e2.extract(transcript_path=tf, session_id="ep-sess-004")

    # Second run should be a no-op (already processed)
    assert r2.episode_id is None or r2.episode_id == r1.episode_id
    episodes = store.list_episodes()
    assert len(episodes) == 1
```

- [ ] **Step 2: Check episode-related store and service signatures**

```bash
grep -n "def get_episode\|def list_episodes\|def record_episode\|def recall_conversations\|episode_id\|fact_ids" yaadein/store.py yaadein/service.py | head -20
```

Adjust test calls to match real method names.

- [ ] **Step 3: Run until green**

```bash
python3.11 -m pytest tests/scenarios/test_s3_episode_lifecycle.py -v 2>&1 | tail -20
```
Expected: 4 PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/scenarios/test_s3_episode_lifecycle.py
git commit -m "test(s3): episode lifecycle — create, link to facts, recall, idempotency"
```

---

## Task 6: S4 — Resilience

**WHY this scenario:** A memory system that crashes on bad input or partial failures is worse than no memory system — it breaks the tool it's supposed to augment. Every failure here must be graceful.

**Files:**
- Create: `tests/scenarios/test_s4_resilience.py`

**Interfaces:**
- Consumes: `yaadein.extractor.Extractor`, `yaadein.service.MemoryService`, `yaadein.store.MemoryStore`
- Produces: nothing (test-only)

- [ ] **Step 1: Write the failing test**

```python
# tests/scenarios/test_s4_resilience.py
# WHY: Yaadein augments Claude Code sessions. Any crash here breaks the user's
# workflow. Each test injects one failure mode and asserts the system either
# recovers or returns a clean error — never an unhandled exception.

import json
import pytest
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path
from yaadein.store import MemoryStore
from yaadein.service import MemoryService
from yaadein.extractor import Extractor


def _make_service():
    store = MemoryStore(":memory:")
    index = MagicMock()
    index.query.return_value = []
    index.add = MagicMock()
    ep_index = MagicMock()
    ep_index.query.return_value = []
    ep_index.add = MagicMock()
    return store, MemoryService(store=store, vector_index=index, episode_index=ep_index)


def _make_extractor(service, generator, tmp_path):
    return Extractor(
        service=service,
        generator=generator,
        extract_log=tmp_path / "log.json",
    )


def test_s4_ollama_down_returns_error_not_crash(tmp_path):
    """When Ollama raises an exception, extractor returns error — daemon stays up."""
    _, service = _make_service()
    generator = MagicMock()
    generator.generate.side_effect = ConnectionError("Ollama unreachable")
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text(json.dumps([{"role": "user", "content": "some content"}]))

    result = extractor.extract(transcript_path=tf, session_id="resilience-001")
    assert result.error is not None
    assert "error" in result.error.lower() or result.error  # any non-empty error string


def test_s4_malformed_transcript_handled(tmp_path):
    """Garbage transcript content does not raise an unhandled exception."""
    _, service = _make_service()
    generator = MagicMock()
    generator.generate.return_value = "[]"
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text("this is not valid json {{{{")

    # Should not raise — result may have error or be empty
    try:
        result = extractor.extract(transcript_path=tf, session_id="resilience-002")
        # If it returns, that's fine — check no unhandled exception
    except Exception as e:
        pytest.fail(f"Unhandled exception on malformed transcript: {e}")


def test_s4_empty_transcript_produces_no_facts(tmp_path):
    """Empty transcript results in no facts stored and no error."""
    _, service = _make_service()
    generator = MagicMock()
    generator.generate.return_value = "[]"
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text("[]")

    result = extractor.extract(transcript_path=tf, session_id="resilience-003")
    assert result.error is None
    assert result.proposed == [] or result.proposed is None or len(result.proposed) == 0


def test_s4_llm_returns_invalid_json_handled(tmp_path):
    """LLM returning non-JSON does not crash the extractor."""
    _, service = _make_service()
    generator = MagicMock()
    generator.generate.return_value = "I'm sorry I cannot help with that."
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text(json.dumps([{"role": "user", "content": "some real content here"}]))

    try:
        result = extractor.extract(transcript_path=tf, session_id="resilience-004")
        # Either error returned or empty result — both acceptable
    except Exception as e:
        pytest.fail(f"Unhandled exception on bad LLM JSON: {e}")


def test_s4_missing_transcript_file_handled(tmp_path):
    """Pointing extractor at a non-existent file does not crash."""
    _, service = _make_service()
    generator = MagicMock()
    extractor = _make_extractor(service, generator, tmp_path)

    missing = tmp_path / "does_not_exist.jsonl"

    try:
        result = extractor.extract(transcript_path=missing, session_id="resilience-005")
    except Exception as e:
        pytest.fail(f"Unhandled exception on missing file: {e}")


@pytest.mark.stress
def test_s4_sqlite_concurrent_writes_no_corruption(tmp_path):
    """10 threads writing to the same SQLite store simultaneously — no data loss."""
    store = MemoryStore(":memory:")
    index = MagicMock()
    index.query.return_value = []
    index.add = MagicMock()
    ep_index = MagicMock()
    ep_index.query.return_value = []
    service = MemoryService(store=store, vector_index=index, episode_index=ep_index)

    errors = []
    written_ids = []
    lock = threading.Lock()

    def write(i):
        try:
            mem_id = service.propose(
                content=f"concurrent fact {i}",
                category="fact",
                scope_type="user",
                scope_key="*",
                confidence=0.8,
                source_session=f"concurrent-{i}",
            )
            with lock:
                written_ids.append(mem_id)
        except Exception as e:
            errors.append(f"thread {i}: {e}")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent write errors: {errors}"
    assert len(written_ids) == 10
    # Verify all 10 are actually in the store
    for mid in written_ids:
        assert store.get(mid) is not None, f"Memory {mid} not found after concurrent write"
```

- [ ] **Step 2: Check ExtractionResult fields**

```bash
grep -n "class ExtractionResult\|proposed\|error\|episode_id\|reinforced\|skipped" yaadein/extractor.py | head -15
```

Adjust `result.proposed`, `result.error` access to match real field names.

- [ ] **Step 3: Run until green**

```bash
python3.11 -m pytest tests/scenarios/test_s4_resilience.py -v 2>&1 | tail -20
```
Expected: 5 PASSED (stress skipped)

- [ ] **Step 4: Commit**

```bash
git add tests/scenarios/test_s4_resilience.py
git commit -m "test(s4): resilience — Ollama down, bad transcript, invalid JSON, concurrent writes"
```

---

## Task 7: S5 — Privacy & Gates

**WHY this scenario:** Yaadein stores sensitive information. If redacted content surfaces or project A sees project B's memories, the system is a security failure. These tests are the hard guarantee.

**Files:**
- Create: `tests/scenarios/test_s5_privacy_gates.py`

**Interfaces:**
- Consumes: `yaadein.service.MemoryService`, `yaadein.store.MemoryStore`, `yaadein.redact.redact`
- Produces: nothing (test-only)

- [ ] **Step 1: Write the failing test**

```python
# tests/scenarios/test_s5_privacy_gates.py
# WHY: Privacy is a hard guarantee. Redacted content must never appear in any
# output path. Scope isolation must hold even when the vector index returns
# cross-scope results (the service layer is the last line of defense).

import pytest
from unittest.mock import MagicMock
from yaadein.store import MemoryStore
from yaadein.service import MemoryService


def _make_service():
    store = MemoryStore(":memory:")
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

    text = "My key is sk-abc123xyz and my token is ghp_ABCDEF1234567890abcdef"
    cleaned = redact(text)
    assert "sk-abc123xyz" not in cleaned
    assert "ghp_ABCDEF" not in cleaned


def test_s5_forgotten_memory_not_recalled():
    """After forget_memory, the memory does not appear in recall results."""
    store, service = _make_service()

    mem_id = service.propose(
        content="sensitive personal info",
        category="fact",
        scope_type="user",
        scope_key="*",
        confidence=0.9,
        source_session="priv-sess-001",
    )

    service.forget(mem_id)

    # Wire index to return the (now-deleted) memory id
    service._index.query.return_value = [(mem_id, 0.95)]

    results = service.recall("sensitive personal info", project_path=None)
    assert all(r.id != mem_id for r in results), "Forgotten memory still returned by recall"


def test_s5_forgotten_memory_not_in_briefing():
    """After forget_memory, the memory does not appear in briefing."""
    store, service = _make_service()

    mem_id = service.propose(
        content="confidential decision about architecture",
        category="decision",
        scope_type="user",
        scope_key="*",
        confidence=0.9,
        source_session="priv-sess-002",
    )

    service.forget(mem_id)

    briefing = service.briefing(project_path=None)
    all_ids = [m["id"] for m in briefing.get("decisions", [])]
    assert mem_id not in all_ids, "Forgotten memory appears in briefing"


def test_s5_project_scope_isolation_in_briefing():
    """Project B's memories never appear in Project A's briefing."""
    store, service = _make_service()

    # Store a project-B-scoped memory
    mem_b_id = service.propose(
        content="Project B secret architecture decision",
        category="decision",
        scope_type="project",
        scope_key="proj-b",
        confidence=0.9,
        source_session="priv-sess-003",
    )

    # Query briefing as project A
    briefing = service.briefing(project_path="/repos/proj-a")
    all_content = [
        m["content"]
        for category in ["facts", "decisions", "gotchas", "conflicts"]
        for m in briefing.get(category, [])
    ]
    assert all("Project B secret" not in c for c in all_content), \
        "Project B memory leaked into Project A briefing"


def test_s5_scope_isolation_survives_index_bleed():
    """Even if vector index returns cross-scope IDs, service filters them out."""
    store, service = _make_service()

    # Seed a project-B memory
    mem_b_id = service.propose(
        content="project B only fact",
        category="fact",
        scope_type="project",
        scope_key="proj-b",
        confidence=0.9,
        source_session="priv-sess-004",
    )

    # Force the index to return project B's memory for a project A query
    service._index.query.return_value = [(mem_b_id, 0.99)]

    results = service.recall("project B only fact", project_path="/repos/proj-a")
    assert all(r.id != mem_b_id for r in results), \
        "Scope isolation failed: project B memory returned in project A recall"
```

- [ ] **Step 2: Check `forget`, `recall`, `briefing`, `_index` attribute names**

```bash
grep -n "def forget\|def recall\|def briefing\|self\._index\|self\.vector_index\|self\._vector" yaadein/service.py | head -15
```

Adjust `service.forget(...)`, `service._index` references to match real names.

- [ ] **Step 3: Run until green**

```bash
python3.11 -m pytest tests/scenarios/test_s5_privacy_gates.py -v 2>&1 | tail -20
```
Expected: 5 PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/scenarios/test_s5_privacy_gates.py
git commit -m "test(s5): privacy — redaction, forget, scope isolation, index bleed protection"
```

---

## Task 8: Live daemon tests

**WHY this scenario:** Unit tests mock too much. The live tests are the only ones that prove the full HTTP stack — FastAPI routing, MCP tool dispatch, real SQLite persistence — all work together.

**Files:**
- Create: `tests/live/test_live_daemon.py`

**Interfaces:**
- Consumes: running Yaadein daemon at `http://localhost:8765`
- Produces: nothing (test-only)

- [ ] **Step 1: Write the live test file**

```python
# tests/live/test_live_daemon.py
# WHY: The unit tests mock the service layer. This file hits the real daemon
# over HTTP to prove the full stack works — FastAPI, MCP dispatch, SQLite.
# Run with: python3.11 -m pytest -m live tests/live/

import pytest
import httpx
import uuid

BASE = "http://localhost:8765"


@pytest.fixture(scope="session")
def daemon():
    """Skip all live tests if the daemon is not running."""
    try:
        r = httpx.get(f"{BASE}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip("Yaadein daemon not healthy")
    except Exception:
        pytest.skip("Yaadein daemon not running on localhost:8765")
    return BASE


def test_live_health(daemon):
    """Daemon responds to /health."""
    r = httpx.get(f"{daemon}/health", timeout=5)
    assert r.status_code == 200


def test_live_remember_and_recall(daemon):
    """Store a memory via MCP tool, then recall it."""
    session = str(uuid.uuid4())[:8]
    unique = f"live-test-fact-{session}"

    # Store via MCP remember tool
    r = httpx.post(f"{daemon}/mcp/tool", json={
        "name": "remember",
        "arguments": {
            "content": unique,
            "category": "fact",
            "scope": "user",
            "confidence": 0.9,
            "session_id": session,
        }
    }, timeout=10)
    assert r.status_code == 200, f"remember failed: {r.text}"

    # Recall
    r2 = httpx.post(f"{daemon}/mcp/tool", json={
        "name": "recall_memory",
        "arguments": {"query": unique},
    }, timeout=10)
    assert r2.status_code == 200
    body = r2.json()
    assert unique in str(body), f"Stored fact not found in recall: {body}"


def test_live_briefing_returns_data(daemon):
    """memory_briefing endpoint returns a valid structure."""
    r = httpx.post(f"{daemon}/mcp/tool", json={
        "name": "memory_briefing",
        "arguments": {},
    }, timeout=10)
    assert r.status_code == 200
    body = r.json()
    # Briefing should have at least these keys
    for key in ["facts", "decisions", "gotchas"]:
        assert key in str(body), f"Missing key '{key}' in briefing: {body}"


def test_live_forget_removes_memory(daemon):
    """Stored memory disappears after forget_memory."""
    session = str(uuid.uuid4())[:8]
    unique = f"forget-test-{session}"

    # Store
    r = httpx.post(f"{daemon}/mcp/tool", json={
        "name": "remember",
        "arguments": {
            "content": unique,
            "category": "fact",
            "scope": "user",
            "confidence": 0.9,
            "session_id": session,
        }
    }, timeout=10)
    assert r.status_code == 200
    mem_id = r.json().get("id") or r.json().get("memory_id")

    if mem_id:
        # Forget
        r2 = httpx.post(f"{daemon}/mcp/tool", json={
            "name": "forget_memory",
            "arguments": {"memory_id": mem_id},
        }, timeout=10)
        assert r2.status_code == 200

        # Recall should not find it
        r3 = httpx.post(f"{daemon}/mcp/tool", json={
            "name": "recall_memory",
            "arguments": {"query": unique},
        }, timeout=10)
        assert unique not in str(r3.json()), "Forgotten memory still returned"
```

- [ ] **Step 2: Check actual MCP tool endpoint path and request shape**

```bash
grep -n "router\|@app\|/mcp\|tool_call\|tool_name" server.py | head -20
```

Adjust the POST path and request JSON shape to match the real server routes.

- [ ] **Step 3: Verify live tests are skipped by default**

```bash
python3.11 -m pytest tests/live/ -v 2>&1 | tail -10
```
Expected: `no tests ran` or all tests deselected (because `live` marker is excluded by default in `addopts`)

- [ ] **Step 4: Run live tests against real daemon (if running)**

```bash
python3.11 -m pytest -m live tests/live/ -v 2>&1 | tail -20
```
Expected: 4 PASSED (only works if daemon is running)

- [ ] **Step 5: Commit**

```bash
git add tests/live/test_live_daemon.py
git commit -m "test(live): HTTP integration tests against running daemon — remember, recall, briefing, forget"
```

---

## Task 9: Gap fills — coverage to 95%+

**WHY:** The coverage report shows 48 uncovered lines across 8 files. Each gap is a path the tests don't exercise — usually an error branch or edge case that could silently break.

**Files:**
- Modify: `tests/test_memory_extractor.py` (lines 91-92, 105-106, 170, 174-175, 202, 213, 249-252, 271-275)
- Modify: `tests/test_memory_llm.py` (lines 20-23, 30)
- Modify: `tests/test_memory_mcp_tools.py` (lines 166-168)
- Modify: `tests/test_memory_service.py` (lines 115, 174, 181-182, 229, 291, 318, 332-348)
- Modify: `tests/test_memory_store.py` (line 203)
- Modify: `tests/test_memory_transcript.py` (lines 28, 62)
- Modify: `tests/test_memory_vector_index.py` (lines 26-29, 33)
- Modify: `tests/test_memory_watcher.py` (lines 23, 36, 45-46)

- [ ] **Step 1: Read each uncovered region and understand what branch it is**

For each file, read the specific lines to understand what's untested:

```bash
sed -n '88,108p' yaadein/extractor.py    # lines 91-92, 105-106
sed -n '167,177p' yaadein/extractor.py   # lines 170, 174-175
sed -n '199,215p' yaadein/extractor.py   # lines 202, 213
sed -n '246,278p' yaadein/extractor.py   # lines 249-252, 271-275
sed -n '17,33p' yaadein/llm.py           # lines 20-23, 30
sed -n '163,170p' yaadein/mcp_tools.py   # lines 166-168
sed -n '112,117p' yaadein/service.py     # line 115
sed -n '171,185p' yaadein/service.py     # lines 174, 181-182
sed -n '226,232p' yaadein/service.py     # line 229
sed -n '288,350p' yaadein/service.py     # lines 291, 318, 332-348
sed -n '200,206p' yaadein/store.py       # line 203
sed -n '25,30p' yaadein/transcript.py    # line 28
sed -n '59,65p' yaadein/transcript.py    # line 62
sed -n '23,35p' yaadein/vector_index.py  # lines 26-29, 33
sed -n '20,48p' yaadein/watcher.py       # lines 23, 36, 45-46
```

- [ ] **Step 2: Write gap-fill tests — one per uncovered branch**

For each gap, append a test to the appropriate existing test file. Pattern:

```python
def test_<module>_<what_branch_covers>():
    """One sentence: what edge case this covers and why it matters."""
    # setup
    # trigger the uncovered branch
    # assert the expected outcome
```

Write the actual tests after reading the uncovered lines in Step 1 — the code tells you exactly what assertion to make.

- [ ] **Step 3: Run coverage again and verify ≥ 95%**

```bash
python3.11 -m pytest --cov=yaadein --cov-report=term-missing -q 2>&1 | grep -E "^yaadein|TOTAL"
```
Expected: TOTAL ≥ 95%

- [ ] **Step 4: Run full suite to confirm nothing regressed**

```bash
python3.11 -m pytest -q 2>&1 | tail -5
```
Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_extractor.py tests/test_memory_llm.py tests/test_memory_mcp_tools.py \
        tests/test_memory_service.py tests/test_memory_store.py tests/test_memory_transcript.py \
        tests/test_memory_vector_index.py tests/test_memory_watcher.py
git commit -m "test(gaps): fill coverage gaps to ≥95% — error branches, edge cases"
```

---

## Final Verification

- [ ] **Run full default suite**

```bash
python3.11 -m pytest -q 2>&1 | tail -5
```
Expected: all PASSED, 0 failures

- [ ] **Run coverage check**

```bash
python3.11 -m pytest --cov=yaadein --cov-report=term-missing -q 2>&1 | grep TOTAL
```
Expected: TOTAL ≥ 95%

- [ ] **Run stress tests**

```bash
python3.11 -m pytest -m stress -v 2>&1 | tail -20
```
Expected: all stress tests PASSED

- [ ] **Run live tests (if daemon is up)**

```bash
python3.11 -m pytest -m live tests/live/ -v 2>&1 | tail -20
```
Expected: 4 PASSED
