import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"YAADEIN_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"YAADEIN_{name}", str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"YAADEIN_{name}", str(default)))


# ── Server ────────────────────────────────────────────────
SERVER_HOST = _env("HOST", "127.0.0.1")
SERVER_PORT = _env_int("PORT", 8899)

# ── Local models (Ollama) ─────────────────────────────────
OLLAMA_BASE_URL = _env("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = _env("LLM_MODEL", "gemma4")

# ── Memory store ──────────────────────────────────────────
MEMORY_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "memory_store")))
MEMORY_DB_PATH = MEMORY_DIR / "memories.db"
MEMORY_CHROMA_DIR = MEMORY_DIR / "chroma_db"
MEMORY_COLLECTION = "yaadein_memories"
MEMORY_TOP_K = _env_int("TOP_K", 5)
MEMORY_KEYWORD_BONUS = 0.1
MEMORY_BRIEFING_LIMITS = {"facts": 10, "decisions": 5, "gotchas": 5}

# ── Extraction pipeline ───────────────────────────────────
MEMORY_MAX_PER_SESSION = _env_int("MAX_PER_SESSION", 5)
MEMORY_CONFIDENCE_FLOOR = _env_float("CONFIDENCE_FLOOR", 0.6)
MEMORY_REINFORCE_THRESHOLD = _env_float("REINFORCE_THRESHOLD", 0.9)
# fits gemma4's 8192-token window with room for the prompt
# (code-heavy text runs ~3 chars/token)
MEMORY_TRANSCRIPT_MAX_CHARS = _env_int("TRANSCRIPT_MAX_CHARS", 16000)
MEMORY_EXTRACT_LOG = MEMORY_DIR / ".extracted.json"

# ── Transcript watcher (near-real-time extraction) ────────
# Every interval, re-mine transcripts modified within 2x the interval.
# Safe to re-run: unchanged files are hash-skipped, known facts reinforce.
# Set YAADEIN_WATCH_INTERVAL=0 to disable the watcher entirely.
MEMORY_WATCH_INTERVAL_SECONDS = _env_int("WATCH_INTERVAL", 30)

# Each source: where a harness keeps transcripts, how to find them (glob),
# what to label memories with (harness), and which parser understands them
# (format — must exist in yaadein.transcript.PARSERS, else the source is
# skipped with a warning until a parser is contributed).
# Override the whole list with YAADEIN_WATCH_SOURCES='[{...}, ...]' (JSON).
_DEFAULT_WATCH_SOURCES = [
    {
        "root": str(Path.home() / ".claude" / "projects"),
        "glob": "*/*.jsonl",
        "harness": "claude-code",
        "format": "claude-jsonl",
    },
    {
        "root": str(
            Path.home() / "Library" / "Application Support" / "Kiro"
            / "User" / "globalStorage" / "kiro.kiroagent" / "sessions"
        ),
        "glob": "*.json",
        "harness": "kiro",
        "format": "kiro-sessions",  # parser pending — source auto-activates once added
    },
]
MEMORY_WATCH_SOURCES = (
    json.loads(os.environ["YAADEIN_WATCH_SOURCES"])
    if "YAADEIN_WATCH_SOURCES" in os.environ
    else _DEFAULT_WATCH_SOURCES
)
