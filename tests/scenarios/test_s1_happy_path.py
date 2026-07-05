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

    The fake index remembers every (id, text) pair passed to add() and, on
    query(), scores each stored text against the query by word overlap —
    an exact/near-exact match scores 1.0, unrelated text scores low. This
    keeps recall() working without a real embedding model, while still
    letting genuinely distinct facts (e.g. "User fact number 0" vs
    "User fact number 1") be told apart, the way a real embedding model
    would, instead of every stored id colliding at similarity=1.0.
    """
    from yaadein.store import MemoryStore
    from yaadein.service import MemoryService

    store = MemoryStore(tmp_path / "memories.db")

    # Tracking index: records (id, text) on add(), scores by word overlap on query()
    stored: list = []  # list of (memory_id, text)

    def _score(query: str, text: str) -> float:
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        if not q_words or not t_words:
            return 0.0
        overlap = len(q_words & t_words)
        return overlap / len(q_words | t_words)

    index = MagicMock()
    index.add.side_effect = lambda memory_id, text: stored.append((memory_id, text))

    def _query(query: str, top_k: int = 20):
        scored = [(mid, _score(query, text)) for mid, text in stored]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    index.query.side_effect = _query

    episode_index = MagicMock()
    episode_index.query.return_value = []
    episode_index.add = MagicMock()

    service = MemoryService(store=store, vector_index=index, episode_index=episode_index)
    service.store = store  # test-only convenience accessor for direct store assertions
    return service


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

    # recall() caps results at MEMORY_TOP_K, so it can't confirm all 20 writes
    # landed. Go straight to the store instead: list() returns every row,
    # unbounded. Each fact's content is distinct ("User fact number 0..19"),
    # so none of them should be similar enough to trigger the extractor's
    # dedup/reinforce path — 20 concurrent extractions should mean 20 rows.
    all_memories = service.store.list()
    fact_memories = [m for m in all_memories if "User fact number" in m.content]
    distinct_contents = {m.content for m in fact_memories}
    assert len(distinct_contents) == 20, (
        f"Expected 20 distinct facts written, got {len(distinct_contents)}: "
        f"{sorted(distinct_contents)}"
    )
    assert len(fact_memories) == 20, (
        f"Expected 20 memory rows (no data loss, no unexpected merging), "
        f"got {len(fact_memories)}"
    )
