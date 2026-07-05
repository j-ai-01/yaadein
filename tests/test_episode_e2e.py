"""End-to-end: transcript -> extraction -> episode searchable -> drillable -> briefed."""
import json
import math

from yaadein.extractor import Extractor
from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["kyun", "provenance", "deploy", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class SequencedGenerator:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("no more canned responses")
        return self._responses.pop(0)


def test_full_episode_pipeline(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join([
        json.dumps({"type": "user", "message": {"role": "user",
                    "content": "Let's design Kyun - git blame for why."}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Kyun will match code lines to the conversations that created them."}]}}),
    ]))

    facts = json.dumps([{
        "content": "Kyun matches code lines to conversations for provenance",
        "category": "decision", "scope": "user", "confidence": 0.9,
        "evidence_quote": "git blame for why",
    }])
    summary = "Talked about the Kyun provenance design and how it maps code to conversations."

    store = MemoryStore(tmp_path / "memories.db")
    service = MemoryService(
        store=store,
        vector_index=MemoryVectorIndex(tmp_path / "cf", FakeEmbedder(), "e2e_facts"),
        episode_index=MemoryVectorIndex(tmp_path / "ce", FakeEmbedder(), "e2e_eps"),
    )
    extractor = Extractor(service=service, generator=SequencedGenerator([facts, summary]),
                          extract_log=tmp_path / ".extracted.json")

    result = extractor.extract(transcript, session_id="sess-e2e")
    assert result.error is None and result.episode_id

    # searchable by meaning
    hits = service.recall_episodes("kyun provenance discussion")
    assert hits and hits[0]["id"] == result.episode_id

    # drillable to ground truth
    detail = service.read_episode(result.episode_id)
    assert "git blame for why" in detail["excerpt"]
    assert detail["fact_ids"] == result.written

    # surfaced in the briefing
    briefing = service.briefing()
    assert briefing["recent_conversations"][0]["id"] == result.episode_id
