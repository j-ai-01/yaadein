"""Preflight checks that the local Ollama server is reachable before depending on it."""

import httpx
from config import OLLAMA_BASE_URL


def check_ollama_running() -> bool:
    """Return True if Ollama responds at OLLAMA_BASE_URL within a short timeout."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def assert_ollama_running() -> None:
    """Exit the process with a helpful message if Ollama isn't running."""
    if not check_ollama_running():
        raise SystemExit(
            "\nOllama is not running.\n"
            "Start it with: ollama serve\n"
            "Then re-run this command."
        )
