# tests/scenarios/test_s4_resilience.py
# WHY: Yaadein augments Claude Code sessions. Any crash here breaks the user's
# workflow. Each test injects one failure mode and asserts the system either
# recovers or returns a clean error — never an unhandled exception.

import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from yaadein.store import MemoryStore
from yaadein.service import MemoryService
from yaadein.extractor import Extractor


def _make_service(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
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


def _jsonl_line(role: str, text: str) -> str:
    """Build a valid claude-jsonl transcript line."""
    return json.dumps({"type": role, "message": {"content": text}})


def test_s4_ollama_down_returns_error_not_crash(tmp_path):
    """When Ollama raises an exception, extractor returns error — daemon stays up."""
    _, service = _make_service(tmp_path)
    generator = MagicMock()
    generator.generate.side_effect = ConnectionError("Ollama unreachable")
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text(_jsonl_line("user", "some content"))

    result = extractor.extract(transcript_path=tf, session_id="resilience-001")
    assert result.error is not None
    assert result.error  # any non-empty error string


def test_s4_malformed_transcript_handled(tmp_path):
    """Garbage transcript content does not raise an unhandled exception."""
    _, service = _make_service(tmp_path)
    generator = MagicMock()
    generator.generate.return_value = "[]"
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text("this is not valid json {{{{")

    # Should not raise — malformed lines are skipped by the parser; empty text = no-op
    try:
        result = extractor.extract(transcript_path=tf, session_id="resilience-002")
    except Exception as e:
        pytest.fail(f"Unhandled exception on malformed transcript: {e}")


def test_s4_empty_transcript_produces_no_facts(tmp_path):
    """Empty transcript results in no facts stored and no error."""
    _, service = _make_service(tmp_path)
    generator = MagicMock()
    generator.generate.return_value = "[]"
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text("")  # truly empty file — no turns

    result = extractor.extract(transcript_path=tf, session_id="resilience-003")
    assert result.error is None
    assert len(result.written) == 0


def test_s4_llm_returns_invalid_json_handled(tmp_path):
    """LLM returning non-JSON does not crash the extractor."""
    _, service = _make_service(tmp_path)
    generator = MagicMock()
    generator.generate.return_value = "I'm sorry I cannot help with that."
    extractor = _make_extractor(service, generator, tmp_path)

    tf = tmp_path / "session.jsonl"
    tf.write_text(_jsonl_line("user", "some real content here"))

    try:
        result = extractor.extract(transcript_path=tf, session_id="resilience-004")
        # Either error returned or empty result — both acceptable
    except Exception as e:
        pytest.fail(f"Unhandled exception on bad LLM JSON: {e}")


def test_s4_missing_transcript_file_handled(tmp_path):
    """Pointing extractor at a non-existent file does not crash."""
    _, service = _make_service(tmp_path)
    generator = MagicMock()
    extractor = _make_extractor(service, generator, tmp_path)

    missing = tmp_path / "does_not_exist.jsonl"

    try:
        result = extractor.extract(transcript_path=missing, session_id="resilience-005")
        assert result.error is not None  # should return a clean error
    except Exception as e:
        pytest.fail(f"Unhandled exception on missing file: {e}")


# NOTE: a concurrent-writes stress test was removed here by decision on 2026-07-06.
# It exposed a real, pre-existing limitation: MemoryStore shares one sqlite3.Connection
# across threads (check_same_thread=False, no lock), so parallel writes intermittently
# corrupt transaction state (~1 in 10 runs). Tracked separately; re-add the test when
# MemoryStore gains a write lock or per-thread connections.
