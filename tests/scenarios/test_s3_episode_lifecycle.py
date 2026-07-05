# tests/scenarios/test_s3_episode_lifecycle.py
# WHY: Episode lifecycle is: transcript arrives → extractor writes episode with summary
# → episode links to extracted facts → recall_episodes returns it ranked by recency.
# Each test checks one link in that chain.

import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from yaadein.store import MemoryStore
from yaadein.service import MemoryService
from yaadein.extractor import Extractor


# Claude Code JSONL format: one JSON object per line
_USER_TEXT = "We decided to use SQLite for the memory store."
_ASST_TEXT = "Good choice — simple, reliable, no server needed."

TRANSCRIPT = "\n".join([
    json.dumps({"type": "user", "message": {"content": _USER_TEXT}}),
    json.dumps({"type": "assistant", "message": {"content": _ASST_TEXT}}),
])

FACT_RESPONSE = json.dumps([{
    "content": "SQLite chosen for memory store",
    "category": "decision",
    "scope": "user",
    "confidence": 0.95,
    "evidence_quote": _USER_TEXT,
}])

SUMMARY_RESPONSE = "Team decided to use SQLite for the memory store because it is simple and reliable."


def _make_service(db_path: Path):
    """Build a MemoryService with tracking fake vector/episode indices."""
    store = MemoryStore(db_path)

    stored_ids: list = []
    index = MagicMock()
    index.add.side_effect = lambda memory_id, text: stored_ids.append(memory_id)
    index.query.side_effect = lambda query, top_k=20: [(mid, 1.0) for mid in stored_ids]

    episode_ids: list = []
    ep_index = MagicMock()
    ep_index.add.side_effect = lambda ep_id, text: episode_ids.append(ep_id)
    ep_index.query.side_effect = lambda query, top_k=20: [(eid, 0.9) for eid in episode_ids]

    service = MemoryService(store=store, vector_index=index, episode_index=ep_index)
    return store, service


def _make_extractor(service, fact_resp: str, summary_resp: str, extract_log: Path):
    generator = MagicMock()
    # First call = fact extraction, second call = episode summary
    generator.generate.side_effect = [fact_resp, summary_resp]
    return Extractor(service=service, generator=generator, extract_log=extract_log)


def test_s3_episode_created_after_extraction(tmp_path):
    """Extraction creates an episode record in the store."""
    store, service = _make_service(tmp_path / "memories.db")
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-001")

    assert result.error is None, f"Extraction error: {result.error}"
    assert result.episode_id is not None, "Expected episode_id to be set after extraction"
    episode = store.get_episode(result.episode_id)
    assert episode is not None, "Episode not found in store"
    assert "SQLite" in episode.summary


def test_s3_episode_linked_to_facts(tmp_path):
    """Facts extracted in the same session are linked to the episode."""
    store, service = _make_service(tmp_path / "memories.db")
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-002")

    assert result.episode_id is not None
    fact_ids = store.fact_ids_for_episode(result.episode_id)
    assert len(fact_ids) >= 1, "Expected at least one fact linked to the episode"
    for fid in fact_ids:
        memory = store.get(fid)
        assert memory is not None, f"Memory {fid} linked to episode but not found in store"


def test_s3_recall_conversations_returns_episode(tmp_path):
    """recall_episodes finds the episode after extraction."""
    store, service = _make_service(tmp_path / "memories.db")
    extractor = _make_extractor(service, FACT_RESPONSE, SUMMARY_RESPONSE, tmp_path / "log.json")

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)
    result = extractor.extract(transcript_path=tf, session_id="ep-sess-003")

    assert result.episode_id is not None
    episodes = service.recall_episodes("SQLite memory store", project_key=None)
    assert any(ep["id"] == result.episode_id for ep in episodes), (
        f"Expected episode {result.episode_id} in recall_episodes result, got: {episodes}"
    )


def test_s3_duplicate_session_idempotent(tmp_path):
    """Submitting the same transcript twice does not create two episodes."""
    store, service = _make_service(tmp_path / "memories.db")
    extract_log = tmp_path / "log.json"

    tf = tmp_path / "session.jsonl"
    tf.write_text(TRANSCRIPT)

    # First extraction
    gen1 = MagicMock()
    gen1.generate.side_effect = [FACT_RESPONSE, SUMMARY_RESPONSE]
    e1 = Extractor(service=service, generator=gen1, extract_log=extract_log)
    r1 = e1.extract(transcript_path=tf, session_id="ep-sess-004")
    assert r1.error is None

    # Second extraction — same file, same hash → should skip
    gen2 = MagicMock()
    gen2.generate.side_effect = [FACT_RESPONSE, SUMMARY_RESPONSE]
    e2 = Extractor(service=service, generator=gen2, extract_log=extract_log)
    r2 = e2.extract(transcript_path=tf, session_id="ep-sess-004")

    # Second run should be a no-op (already processed)
    assert r2.already_processed or r2.episode_id is None or r2.episode_id == r1.episode_id
    episodes = store.list_episodes()
    assert len(episodes) == 1, f"Expected 1 episode, got {len(episodes)}"
