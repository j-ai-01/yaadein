# tests/test_extract_endpoint.py
from fastapi.testclient import TestClient

import server


def test_extract_queues_background_extraction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        server, "_run_extraction",
        lambda *args: calls.append(args),
    )
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}")
    client = TestClient(server.app)
    resp = client.post("/memory/extract", json={
        "transcript_path": str(transcript),
        "project_path": "/some/repo",
        "session_id": "sess-1",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    assert calls == [(str(transcript), "/some/repo", "sess-1", "claude-code")]


def test_extract_404_for_missing_transcript(tmp_path):
    client = TestClient(server.app)
    resp = client.post("/memory/extract", json={
        "transcript_path": str(tmp_path / "nope.jsonl"),
    })
    assert resp.status_code == 404
    assert "error" in resp.json()
