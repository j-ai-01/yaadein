import math
from pathlib import Path

import pytest

from yaadein.extractor import Extractor
from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.vector_index import MemoryVectorIndex
from utils.ollama_check import check_ollama_running

FIXTURE = Path(__file__).parent / "fixtures" / "transcripts" / "pytest_preference.jsonl"


class FakeEmbedder:
    def embed(self, text):
        vec = [float((hash(w) % 97) + 1) for w in ["a", "b", "c", "d"]]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_service(tmp_path, embedder):
    return MemoryService(
        store=MemoryStore(tmp_path / "memories.db"),
        vector_index=MemoryVectorIndex(
            chroma_dir=tmp_path / "chroma", embedder=embedder,
            collection_name="eval_memories",
        ),
    )


def test_fixture_parses_and_pipeline_runs_with_fake_llm(tmp_path):
    """Always-on smoke: the eval fixture is valid and the pipeline consumes it."""

    class NothingGenerator:
        def generate(self, prompt):
            assert "pytest over unittest" in prompt  # transcript made it to the LLM
            return "[]"

    extractor = Extractor(
        service=make_service(tmp_path, FakeEmbedder()),
        generator=NothingGenerator(),
        extract_log=tmp_path / ".extracted.json",
    )
    result = extractor.extract(FIXTURE)
    assert result.error is None
    assert result.written == []


@pytest.mark.eval
def test_real_llm_extracts_preference_and_gotcha(tmp_path):
    """Quality eval against real Ollama models. Run with: pytest -m eval"""
    if not check_ollama_running():
        pytest.skip("Ollama not running")
    from yaadein.llm import OllamaGenerator
    from yaadein.vector_index import OllamaEmbedder

    extractor = Extractor(
        service=make_service(tmp_path, OllamaEmbedder()),
        generator=OllamaGenerator(),
        extract_log=tmp_path / ".extracted.json",
    )
    result = extractor.extract(FIXTURE)
    assert result.error is None
    contents = [
        extractor._service._store.get(mid).content.lower() for mid in result.written
    ]
    assert any("pytest" in c for c in contents), f"expected a pytest preference in {contents}"
    assert any("aws_region" in c or "staging" in c for c in contents), (
        f"expected the AWS_REGION gotcha in {contents}"
    )
    assert len(result.written) <= 5
