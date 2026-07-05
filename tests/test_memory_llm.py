import sys
import types
from unittest.mock import MagicMock, patch

from yaadein.llm import TextGenerator
from yaadein.types import Candidate


class CannedGenerator:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_canned_generator_satisfies_protocol():
    gen: TextGenerator = CannedGenerator("hello")
    assert gen.generate("anything") == "hello"


def test_candidate_holds_extraction_fields():
    c = Candidate(
        content="User prefers pytest",
        category="preference",
        scope="user",
        confidence=0.9,
        evidence_quote="I prefer pytest",
    )
    assert c.category == "preference"
    assert c.scope == "user"


def test_ollama_generator_wraps_local_llm_without_network():
    """OllamaGenerator builds an Ollama client with configured model/url and delegates generate() to complete().text."""
    fake_ollama_cls = MagicMock()
    fake_instance = fake_ollama_cls.return_value
    fake_instance.complete.return_value = MagicMock(text="generated text")

    fake_module = types.ModuleType("llama_index.llms.ollama")
    fake_module.Ollama = fake_ollama_cls

    with patch.dict(sys.modules, {"llama_index.llms.ollama": fake_module}):
        from yaadein.llm import OllamaGenerator

        generator = OllamaGenerator()
        result = generator.generate("hello prompt")

    fake_ollama_cls.assert_called_once()
    fake_instance.complete.assert_called_once_with("hello prompt")
    assert result == "generated text"
