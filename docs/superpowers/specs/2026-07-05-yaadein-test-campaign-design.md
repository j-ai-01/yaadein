# Yaadein Thorough Test Campaign

**Date:** 2026-07-05  
**Goal:** Ship confidence + learning density. Every test written forces reasoning about the full data flow.

---

## Approach: Scenario-Driven

5 core scenarios trace Yaadein's full data flow. All tests — unit, live, stress — map to one of these scenarios.

| # | Scenario | What it exercises |
|---|----------|-------------------|
| S1 | Happy path session — transcript → extractor → facts stored → briefing returns them | Full pipeline end-to-end |
| S2 | Recall under load — 500 memories stored, query returns right ones ranked correctly | Vector index, scopes, ranking |
| S3 | Episode lifecycle — conversation recorded → episode created → recall_conversations finds it | Episodic memory layer |
| S4 | Resilience — Ollama down, SQLite locked, malformed transcript, daemon restart mid-session | Error handling, retries, recovery |
| S5 | Privacy & gates — redacted content never surfaces, scope isolation between projects | Security correctness |

---

## Test Suite Structure

```
tests/
├── scenarios/                  # scenario-driven tests (new)
│   ├── test_s1_happy_path.py
│   ├── test_s2_recall_load.py
│   ├── test_s3_episode_lifecycle.py
│   ├── test_s4_resilience.py
│   └── test_s5_privacy_gates.py
├── live/                       # hits real running daemon via HTTP (new)
│   └── test_live_daemon.py
└── ... existing 120 tests stay untouched
```

### Test Modes (pytest markers)

| Marker | When | Hits |
|--------|------|------|
| *(default, no marker)* | Always — CI-safe | Mocked, in-memory |
| `live` | When daemon is running on localhost:8765 | Real HTTP |
| `stress` | Explicitly | Concurrent / load variants |

Add to `pytest.ini`:
```ini
markers =
    eval: extraction-quality evals (need local Ollama)
    live: tests that require the Yaadein daemon running on localhost:8765
    stress: load and concurrency tests
addopts = -m "not eval and not live and not stress"
```

---

## Gap Analysis (one-time audit)

Run before writing new tests:
```bash
python3.11 -m pytest --cov=yaadein --cov-report=term-missing -q
```

Produces a checklist of untested lines/branches. Gap fixes go into existing test files as targeted additions — no new files for gap coverage.

---

## Scenario Specs

### S1 — Happy Path Session
**Data flow:** `POST /memory/extract` with a Claude Code transcript → extractor parses facts → `store.add()` persists them → `GET /memory/briefing` returns them.

Tests:
- Unit: mock extractor + store, verify wiring
- Live: send a real transcript fixture to the running daemon, assert facts appear in briefing response
- Stress: send 20 transcripts concurrently, verify no facts are lost

### S2 — Recall Under Load
**Data flow:** bulk insert 500 facts across 3 scopes → `recall_memory` query → verify top-k correctness and scope isolation.

Tests:
- Unit: insert 500 mocked facts, query, assert ranking and scope boundaries
- Stress: 10 concurrent readers + 5 concurrent writers, assert no cross-scope bleed, latency < 200ms

### S3 — Episode Lifecycle
**Data flow:** session starts → transcript arrives → extractor writes episode → `recall_conversations` returns it ranked by recency.

Tests:
- Unit: mock transcript → verify episode record created with correct fact links
- Live: trigger a real extraction, call `recall_conversations`, assert episode appears
- Edge: duplicate session ID sent twice — verify idempotent (no duplicate episode)

### S4 — Resilience
**Data flow:** things go wrong — system must degrade gracefully, not crash.

| Test | Injection | Expected behavior |
|------|-----------|-------------------|
| Ollama unreachable | Mock HTTP 503 | Extractor retries N times, then stores partial result — daemon stays up |
| SQLite concurrent writes | 10 threads simultaneous | All writes succeed, no corruption (WAL mode) |
| Malformed transcript | Garbage JSON / missing fields / 10MB payload | 400 response, no daemon crash |
| Daemon restart mid-extraction | Kill + restart process | Next extraction succeeds, no duplicate facts |
| Memory store full | Exhaust SQLite row limit (mocked) | Graceful error, existing memories intact |

### S5 — Privacy & Gates
**Data flow:** sensitive content flagged by redactor must never appear in any output path.

Tests:
- Redacted fact never returned by `recall_memory`
- Redacted content absent from `memory_briefing`
- Project A's memories not visible when querying with Project B's scope key
- `forget_memory` removes fact from all retrieval paths (store + vector index)

---

## Live Test Setup

```python
# tests/live/test_live_daemon.py
# WHY: Proves the full stack works together — HTTP → service → store → response.
# Run with: python3.11 -m pytest -m live tests/live/

import pytest, httpx

BASE = "http://localhost:8765"

@pytest.fixture(scope="session")
def daemon():
    r = httpx.get(f"{BASE}/health", timeout=2)
    if r.status_code != 200:
        pytest.skip("Yaadein daemon not running")
    return BASE
```

---

## Success Criteria

- [ ] All 5 scenario files written and passing (default mode)
- [ ] Live test suite passes against running daemon
- [ ] Stress tests: S2 recall < 200ms p95, S1 concurrent no data loss
- [ ] Coverage report shows ≥ 90% line coverage on `yaadein/` package
- [ ] S4 resilience: daemon never crashes on any injected failure
- [ ] S5 privacy: zero cross-scope leaks, zero redacted content in outputs
