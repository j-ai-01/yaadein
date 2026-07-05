# tests/live/test_live_daemon.py
# WHY: The unit tests mock the service layer (server.py's MCP dispatch is
# only exercised via TestClient + monkeypatched extraction). This file hits
# the real, running daemon over plain HTTP to prove the full stack -- FastAPI
# routing, the background extraction pipeline, and real SQLite persistence --
# actually works end to end.
#
# Reality check (see server.py): the six memory tools (remember, recall_memory,
# forget_memory, memory_briefing, recall_conversations, read_conversation) are
# only reachable over MCP-over-SSE (GET /sse + POST /messages, the latter
# mounted as a raw ASGI app that owns the response -- see the comment above
# `app.mount("/messages", ...)` in server.py). There is no `/mcp/tool` REST
# endpoint and no plain-JSON way to invoke `remember`/`recall_memory` over
# HTTP. So these tests exercise what IS reachable over plain HTTP:
#   - GET  /health          liveness
#   - POST /memory/extract  queues a real background extraction run
#   - GET  /sse             confirms the MCP transport is mounted and live
# and use the in-process MemoryService (same MEMORY_DB_PATH the daemon reads
# and writes) only to verify/clean up side effects of the HTTP call above --
# it does not stand in for an HTTP endpoint that doesn't exist.
#
# Run with: python3.11 -m pytest -m live tests/live/

import sqlite3
import time
import uuid

import httpx
import pytest

import config

BASE = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"


def _retry_on_locked(fn, attempts=3, backoff=0.2):
    """yaadein/schema.py's connect() sets no WAL mode and no busy_timeout, so
    this in-process call can race the daemon's own SQLite write transactions
    and raise sqlite3.OperationalError: database is locked -- retry a few
    times with a short backoff instead of flaking the test.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except sqlite3.OperationalError:
            if attempt == attempts:
                raise
            time.sleep(backoff)


@pytest.fixture(scope="session")
def daemon():
    """Skip all live tests if the daemon is not reachable on the configured
    host/port (config.SERVER_HOST/SERVER_PORT, default 127.0.0.1:8899)."""
    try:
        r = httpx.get(f"{BASE}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip("Yaadein daemon not healthy")
    except Exception:
        pytest.skip(f"Yaadein daemon not running on {BASE}")
    return BASE


@pytest.mark.live
def test_live_health(daemon):
    """Daemon responds to /health with a 200 and status ok."""
    r = httpx.get(f"{daemon}/health", timeout=5)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.live
def test_live_sse_endpoint_reachable(daemon):
    """GET /sse opens the MCP-over-SSE event stream (proves the MCP server
    is mounted and dispatching, not just that FastAPI is up)."""
    with httpx.stream("GET", f"{daemon}/sse", timeout=5) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


@pytest.mark.live
def test_live_extract_queues_and_runs(daemon, tmp_path):
    """POST /memory/extract accepts a real transcript, returns "queued", and
    the background task runs against the real extraction pipeline (SQLite +
    Chroma) without the request itself erroring out.

    We don't assert on which facts get extracted -- that depends on Ollama
    being reachable and on LLM output, which this test isn't trying to pin
    down. We only assert the HTTP contract (200, status=queued) holds against
    the real daemon, i.e. the full request path (FastAPI -> BackgroundTasks ->
    _run_extraction -> build_extractor) doesn't blow up before task hand-off.
    """
    import json as _json

    tag = uuid.uuid4().hex[:8]
    transcript = tmp_path / f"live-{tag}.jsonl"
    line = {
        "type": "user",
        "message": {
            "role": "user",
            "content": f"live-daemon-test-marker-{tag}: I prefer tabs over spaces.",
        },
    }
    transcript.write_text(_json.dumps(line) + "\n")

    r = httpx.post(
        f"{daemon}/memory/extract",
        json={
            "transcript_path": str(transcript),
            "session_id": f"live-test-{tag}",
        },
        timeout=10,
    )
    assert r.status_code == 200, f"extract failed: {r.text}"
    body = r.json()
    assert body["status"] == "queued"
    assert body["transcript"] == str(transcript)


@pytest.mark.live
def test_live_extract_404_for_missing_transcript(daemon, tmp_path):
    """Real daemon returns 404 for a transcript path that doesn't exist,
    matching the contract covered by unit tests but proven here over HTTP."""
    missing = tmp_path / "does-not-exist.jsonl"
    r = httpx.post(
        f"{daemon}/memory/extract",
        json={"transcript_path": str(missing)},
        timeout=5,
    )
    assert r.status_code == 404
    assert "error" in r.json()


@pytest.mark.live
def test_live_service_roundtrip_against_daemon_store(daemon):
    """Sanity check that the daemon's configured store (MEMORY_DB_PATH) is a
    real, writable SQLite-backed MemoryService reachable in-process -- i.e.
    the same storage layer the running daemon uses for remember/recall/forget
    (invoked over MCP-over-SSE in production, not plain HTTP). Cleans up the
    memory it writes via forget().
    """
    from yaadein.service import get_memory_service

    tag = uuid.uuid4().hex[:8]
    unique = f"live-daemon-store-check-{tag}"

    service = get_memory_service()
    memory = _retry_on_locked(
        lambda: service.remember(content=unique, category="fact")
    )
    try:
        results = service.recall(unique)
        assert unique in str(results), f"Stored fact not found in recall: {results}"
    finally:
        assert _retry_on_locked(lambda: service.forget(memory.id)) is True

    results_after = service.recall(unique)
    assert unique not in str(results_after), "Forgotten memory still returned"
