import json
import math
import threading
import pytest
from yaadein.extractor import Extractor, _parse_candidates
from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class CannedGenerator:
    def __init__(self, response):
        self._response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self._response


def write_transcript(tmp_path, user_text):
    p = tmp_path / "session.jsonl"
    p.write_text(json.dumps(
        {"type": "user", "message": {"role": "user", "content": user_text}}
    ))
    return p


def canned_json(evidence):
    return json.dumps([{
        "content": "User prefers pytest over unittest",
        "category": "preference", "scope": "user",
        "confidence": 0.9, "evidence_quote": evidence,
    }])


def canned_json_multi():
    return json.dumps([
        {
            "content": "User prefers pytest over unittest",
            "category": "preference", "scope": "user",
            "confidence": 0.9, "evidence_quote": "pytest over unittest",
        },
        {
            "content": "Deploys go through the blue pipeline",
            "category": "fact", "scope": "user",
            "confidence": 0.8, "evidence_quote": "through the blue pipeline",
        },
    ])


def make_extractor(tmp_path, generator):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma", embedder=FakeEmbedder(),
        collection_name="test_memories",
    )
    service = MemoryService(store=store, vector_index=index)
    extractor = Extractor(
        service=service, generator=generator,
        extract_log=tmp_path / ".extracted.json",
    )
    return extractor, store, service


def test_end_to_end_writes_proposed_memory_with_provenance(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, store, _ = make_extractor(tmp_path, gen)
    result = extractor.extract(transcript, session_id="sess-9")
    assert len(result.written) == 1
    row = store.get(result.written[0])
    assert row.status == "proposed"
    assert row.source_session == "sess-9"
    assert row.source_harness == "claude-code"


def test_redaction_happens_before_llm_sees_transcript(tmp_path):
    transcript = write_transcript(tmp_path, "my key is AKIAIOSFODNN7EXAMPLE ok")
    gen = CannedGenerator("[]")
    extractor, _, _ = make_extractor(tmp_path, gen)
    extractor.extract(transcript)
    assert "AKIAIOSFODNN7EXAMPLE" not in gen.prompts[0]


def test_second_run_is_idempotent(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, store, _ = make_extractor(tmp_path, gen)
    extractor.extract(transcript)
    second = extractor.extract(transcript)
    assert second.already_processed is True
    assert len(store.list()) == 1


def test_near_duplicate_reinforces_instead_of_writing(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, store, service = make_extractor(tmp_path, gen)
    existing = service.propose(
        content="User prefers pytest over unittest",
        category="preference", scope_type="user", scope_key="*", confidence=0.7,
    )
    result = extractor.extract(transcript)
    assert result.reinforced == [existing.id]
    assert result.written == []
    assert len(store.list()) == 1
    assert store.get(existing.id).confidence == pytest.approx(0.8)


def test_generator_failure_is_returned_and_retryable(tmp_path):
    class ExplodingGenerator:
        def generate(self, prompt):
            raise RuntimeError("model gone")

    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    extractor, store, _ = make_extractor(tmp_path, ExplodingGenerator())
    result = extractor.extract(transcript)
    assert result.error is not None
    assert store.list() == []
    retry = extractor.extract(transcript)  # not marked processed
    assert retry.already_processed is False


def test_parse_candidates_tolerates_prose_around_json():
    raw = 'Sure! Here you go:\n[{"content": "User prefers pytest over unittest", "category": "preference", "scope": "user", "confidence": 0.9, "evidence_quote": "pytest"}]\nHope that helps.'
    assert len(_parse_candidates(raw)) == 1


def test_parse_candidates_returns_empty_on_garbage():
    assert _parse_candidates("no json here") is None
    assert _parse_candidates('[{"content": 42}]') == []


def test_unparseable_llm_output_is_returned_and_retryable(tmp_path):
    gen = CannedGenerator("Sorry, I can't help with that right now.")
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    extractor, store, _ = make_extractor(tmp_path, gen)
    result = extractor.extract(transcript)
    assert result.error is not None
    assert store.list() == []
    retry = extractor.extract(transcript)  # not marked processed
    assert retry.already_processed is False


def test_embedder_failure_mid_batch_is_returned_and_retryable(tmp_path):
    class CountingEmbedder:
        def __init__(self):
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("embedding backend unavailable")
            return FakeEmbedder().embed(text)

    transcript = write_transcript(
        tmp_path,
        "I prefer pytest over unittest, always. Deploys go through the blue pipeline.",
    )
    gen = CannedGenerator(canned_json_multi())
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma", embedder=CountingEmbedder(),
        collection_name="test_memories",
    )
    service = MemoryService(store=store, vector_index=index)
    extractor = Extractor(
        service=service, generator=gen, extract_log=tmp_path / ".extracted.json",
    )

    result = extractor.extract(transcript)

    assert result.error is not None
    # whatever was already written before the failure stays recorded in the result
    assert len(result.written) == 1
    retry = extractor.extract(transcript)  # not marked processed
    assert retry.already_processed is False


def test_extract_survives_cross_thread_service_use(tmp_path):
    """The service's singleton SQLite connection is created on the main thread but
    extraction runs on a background threadpool thread (see mcp_server._run_extraction).
    Without check_same_thread=False this raises sqlite3.ProgrammingError."""
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, store, service = make_extractor(tmp_path, gen)

    # Touch the service on the main thread first, mirroring get_memory_service()'s
    # singleton being constructed on the main thread before background use.
    service.propose(
        content="warm up the connection", category="fact",
        scope_type="user", scope_key="*", confidence=0.5,
    )

    results = []

    def run():
        results.append(extractor.extract(transcript))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert len(results) == 1
    result = results[0]
    assert result.error is None
    assert result.written or result.reinforced


def test_bookmark_second_pass_mines_only_new_turns(tmp_path):
    """The bookmark: a grown transcript re-mines ONLY the added turns."""
    transcript = tmp_path / "session.jsonl"
    turn_a = json.dumps({"type": "user", "message": {"role": "user", "content": "I prefer pytest over unittest, always."}})
    turn_b = json.dumps({"type": "user", "message": {"role": "user", "content": "Also we deploy with the blue pipeline."}})
    transcript.write_text(turn_a)

    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, store, _ = make_extractor(tmp_path, gen)
    extractor.extract(transcript)
    assert "pytest over unittest" in gen.prompts[0]

    transcript.write_text(turn_a + "\n" + turn_b)  # transcript grows

    class AssertingGenerator:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            assert "blue pipeline" in prompt, "new turn must be visible"
            assert "pytest over unittest" not in prompt, "old turn must NOT be re-read"
            return "[]"

    extractor._generator = AssertingGenerator()
    result = extractor.extract(transcript)
    assert result.error is None
    assert len(store.list()) == 1  # nothing duplicated


def test_bookmark_unchanged_transcript_still_skips(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, _, _ = make_extractor(tmp_path, gen)
    extractor.extract(transcript)
    assert extractor.extract(transcript).already_processed is True


def test_bookmark_tolerates_legacy_bare_hash_log(tmp_path):
    from utils.file_hash import file_hash
    from utils.ingest_tracker import save_ingested

    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    log = tmp_path / ".extracted.json"
    save_ingested({str(transcript): file_hash(transcript)}, log)  # legacy format
    gen = CannedGenerator(canned_json("I prefer pytest over unittest"))
    extractor, _, _ = make_extractor(tmp_path, gen)
    assert extractor.extract(transcript).already_processed is True


class SequencedGenerator:
    """Returns queued responses in order; raises when exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("no more canned responses")
        return self._responses.pop(0)


def make_episodic_extractor(tmp_path, generator):
    store = MemoryStore(tmp_path / "memories.db")
    service = MemoryService(
        store=store,
        vector_index=MemoryVectorIndex(tmp_path / "cf", FakeEmbedder(), "t_facts"),
        episode_index=MemoryVectorIndex(tmp_path / "ce", FakeEmbedder(), "t_eps"),
    )
    extractor = Extractor(service=service, generator=generator,
                          extract_log=tmp_path / ".extracted.json")
    return extractor, store, service


def test_pass_creates_episode_with_stamped_facts(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = SequencedGenerator([
        canned_json("I prefer pytest over unittest"),
        "We discussed testing preferences. Jai prefers pytest.",
    ])
    extractor, store, _ = make_episodic_extractor(tmp_path, gen)
    result = extractor.extract(transcript, session_id="sess-9")
    assert result.episode_id and result.episode_id.startswith("ep_")
    episode = store.get_episode(result.episode_id)
    assert episode.summary.startswith("We discussed testing")
    assert "pytest over unittest" in episode.excerpt  # redacted excerpt of window
    assert store.get(result.written[0]).episode_id == result.episode_id
    assert store.fact_ids_for_episode(result.episode_id) == result.written


def test_zero_fact_pass_still_creates_episode(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = SequencedGenerator(["[]", "Small talk about testing tools."])
    extractor, store, _ = make_episodic_extractor(tmp_path, gen)
    result = extractor.extract(transcript)
    assert result.written == []
    assert result.episode_id is not None
    assert store.get_episode(result.episode_id) is not None


def test_summary_failure_aborts_pass_before_any_writes(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = SequencedGenerator([canned_json("I prefer pytest over unittest")])  # no 2nd response
    extractor, store, _ = make_episodic_extractor(tmp_path, gen)
    result = extractor.extract(transcript)
    assert result.error is not None
    assert store.list() == [] and store.list_episodes() == []  # R9.1: nothing written
    # the transcript was never marked processed:
    gen2 = SequencedGenerator([canned_json("I prefer pytest over unittest"), "Summary."])
    extractor._generator = gen2
    assert extractor.extract(transcript).already_processed is False


def test_episode_excerpt_is_redacted_and_capped(tmp_path):
    from config import MEMORY_EPISODE_EXCERPT_MAX_CHARS

    transcript = write_transcript(
        tmp_path, "my key is AKIAIOSFODNN7EXAMPLE, please don't leak it"
    )
    gen = SequencedGenerator(["[]", "Talked about credentials handling."])
    extractor, store, _ = make_episodic_extractor(tmp_path, gen)
    result = extractor.extract(transcript)
    episode = store.get_episode(result.episode_id)
    assert "AKIAIOSFODNN7EXAMPLE" not in episode.excerpt
    assert len(episode.excerpt) <= MEMORY_EPISODE_EXCERPT_MAX_CHARS


def test_no_episode_index_skips_summary_call_entirely(tmp_path):
    transcript = write_transcript(tmp_path, "I prefer pytest over unittest, always.")
    gen = SequencedGenerator([canned_json("I prefer pytest over unittest")])  # ONE response only
    extractor, store, _ = make_extractor(tmp_path, gen)  # existing helper: no episode index
    result = extractor.extract(transcript)
    assert result.error is None and result.episode_id is None
    assert len(gen.prompts) == 1  # summary LLM call never happened
    assert store.get(result.written[0]).episode_id is None
