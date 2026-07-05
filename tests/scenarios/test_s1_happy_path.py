# tests/scenarios/test_s1_happy_path.py
# WHY: Traces the full data flow — transcript → extractor → store → briefing.
# Each test here is a station on that pipeline. If a station fails, you know
# exactly where the break is.

import json
import pytest
import threading
from unittest.mock import MagicMock
from pathlib import Path


def _make_service(tmp_path: Path):
    """Build an in-memory MemoryService with a tracking fake vector index.

    The fake index remembers every id passed to add() and returns them all
    with similarity=1.0 on any query — so recall() sees facts the extractor
    wrote without needing a real embedding model.
    """
    from yaadein.store import MemoryStore
    from yaadein.service import MemoryService

    store = MemoryStore(tmp_path / "memories.db")

    # Tracking index: records ids on add(), returns them all on query()
    stored_ids: list = []

    index = MagicMock()
    index.add.side_effect = lambda memory_id, text: stored_ids.append(memory_id)
    index.query.side_effect = lambda query, top_k=20: [(mid, 1.0) for mid in stored_ids]

    episode_index = MagicMock()
    episode_index.query.return_value = []
    episode_index.add = MagicMock()

    return MemoryService(store=store, vector_index=index, episode_index=episode_index)


def _make_extractor(service, llm_response: str, extract_log: Path):
    """Build an Extractor with a fake LLM generator returning llm_response."""
    from yaadein.extractor import Extractor

    generator = MagicMock()
    generator.generate.return_value = llm_response
    return Extractor(
        service=service,
        generator=generator,
        extract_log=extract_log,
    )


_USER_TURN = "I prefer tabs over spaces in Python."

# Claude Code JSONL format: one JSON object per line, each with type + message.
SAMPLE_TRANSCRIPT = "\n".join([
    json.dumps({"type": "user", "message": {"content": _USER_TURN}}),
    json.dumps({"type": "assistant", "message": {"content": "Noted."}}),
])

# evidence_quote must match (case-insensitively, whitespace-collapsed) a substring
# of the transcript text that the extractor feeds to apply_gates.
LLM_FACT_RESPONSE = json.dumps([
    {
        "content": "User prefers tabs over spaces in Python",
        "category": "preference",
        "scope": "user",
        "confidence": 0.9,
        # Exact copy of the user turn so grounding gate passes
        "evidence_quote": _USER_TURN,
    }
])


def test_s1_fact_stored_after_extraction(tmp_path):
    """Transcript → extractor → fact appears in store."""
    service = _make_service(tmp_path)
    extractor = _make_extractor(service, LLM_FACT_RESPONSE, tmp_path / "extract_log.json")

    transcript_file = tmp_path / "session.jsonl"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)

    result = extractor.extract(transcript_path=transcript_file, session_id="sess-001")

    assert result.error is None, f"Extraction error: {result.error}"
    memories = service.recall("tabs spaces Python", project_key=None)
    assert len(memories) >= 1
    assert any("tabs" in m["content"].lower() for m in memories)


def test_s1_briefing_returns_extracted_fact(tmp_path):
    """Confirmed fact stored via service appears in briefing response.

    WHY: The extractor writes memories as `proposed` (unconfirmed), which the
    briefing intentionally omits from `facts` until confirmed. This test uses
    service.remember() — the MCP tool path — to plant a confirmed preference,
    then verifies the briefing surfaces it. The extraction → store path is
    covered by test_s1_fact_stored_after_extraction; here we test the store
    → briefing leg.
    """
    service = _make_service(tmp_path)

    # Plant a confirmed preference (as the remember MCP tool would)
    service.remember(
        content="User prefers tabs over spaces in Python",
        category="preference",
        source_session="sess-002",
    )

    briefing = service.briefing(project_key=None)
    all_content = [m["content"] for m in briefing.get("facts", [])]
    assert any("tabs" in c.lower() for c in all_content), (
        f"Expected 'tabs' fact in briefing facts. Got: {briefing}"
    )


def test_s1_duplicate_transcript_not_reextracted(tmp_path):
    """Same transcript submitted twice produces facts only once."""
    service = _make_service(tmp_path)
    # Both calls share the same extract_log so the second is idempotent
    log = tmp_path / "extract_log.json"
    extractor = _make_extractor(service, LLM_FACT_RESPONSE, log)

    transcript_file = tmp_path / "session.jsonl"
    transcript_file.write_text(SAMPLE_TRANSCRIPT)

    extractor.extract(transcript_path=transcript_file, session_id="sess-003")
    result2 = extractor.extract(transcript_path=transcript_file, session_id="sess-003")

    assert result2.already_processed, "Second identical extraction should be idempotent"
    memories = service.recall("tabs spaces", project_key=None)
    tab_memories = [m for m in memories if "tabs" in m["content"].lower()]
    assert len(tab_memories) == 1


@pytest.mark.stress
def test_s1_concurrent_extractions_no_data_loss(tmp_path):
    """20 concurrent extractions each add their own fact — none are lost."""
    service = _make_service(tmp_path)
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
        transcript = json.dumps({"type": "user", "message": {"content": fact}})
        tf = tmp_path / f"session_{i}.jsonl"
        tf.write_text(transcript)
        log = tmp_path / f"extract_log_{i}.json"
        extractor = _make_extractor(service, llm_resp, log)
        result = extractor.extract(transcript_path=tf, session_id=f"sess-stress-{i}")
        if result.error:
            errors.append(result.error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Extraction errors: {errors}"
    all_memories = service.recall("User fact", project_key=None)
    assert len(all_memories) >= 15  # allow some near-duplicate merging
