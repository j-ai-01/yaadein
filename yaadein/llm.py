"""The text-generation backend used to distill transcripts into memory
candidates during extraction (see extractor.py's DISTILL stage).
"""

from typing import Protocol


class TextGenerator(Protocol):
    """Anything that can complete a prompt and return generated text."""

    def generate(self, prompt: str) -> str:
        ...


class OllamaGenerator:
    """TextGenerator backed by a local Ollama LLM (see config.LLM_MODEL)."""

    def __init__(self):
        from llama_index.llms.ollama import Ollama
        from config import LLM_MODEL, OLLAMA_BASE_URL

        self._llm = Ollama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
            request_timeout=120.0, context_window=8192,
        )

    def generate(self, prompt: str) -> str:
        """Send `prompt` to the local LLM and return its completion text."""
        return self._llm.complete(prompt).text
