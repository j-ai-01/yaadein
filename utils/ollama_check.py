import httpx
from config import OLLAMA_BASE_URL


def check_ollama_running() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def assert_ollama_running() -> None:
    if not check_ollama_running():
        raise SystemExit(
            "\nOllama is not running.\n"
            "Start it with: ollama serve\n"
            "Then re-run this command."
        )
