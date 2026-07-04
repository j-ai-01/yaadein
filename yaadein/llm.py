from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class OllamaGenerator:
    def __init__(self):
        from llama_index.llms.ollama import Ollama
        from config import LLM_MODEL, OLLAMA_BASE_URL

        self._llm = Ollama(
            model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
            request_timeout=120.0, context_window=8192,
        )

    def generate(self, prompt: str) -> str:
        return self._llm.complete(prompt).text
