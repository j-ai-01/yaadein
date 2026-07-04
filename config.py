from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Local models (Ollama) ─────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4"

# ── Memory store ──────────────────────────────────────────
MEMORY_DIR = BASE_DIR / "memory_store"
MEMORY_DB_PATH = MEMORY_DIR / "memories.db"
MEMORY_CHROMA_DIR = MEMORY_DIR / "chroma_db"
MEMORY_COLLECTION = "yaadein_memories"
MEMORY_TOP_K = 5
MEMORY_KEYWORD_BONUS = 0.1
MEMORY_BRIEFING_LIMITS = {"facts": 10, "decisions": 5, "gotchas": 5}

# ── Extraction pipeline ───────────────────────────────────
MEMORY_MAX_PER_SESSION = 5
MEMORY_CONFIDENCE_FLOOR = 0.6
MEMORY_REINFORCE_THRESHOLD = 0.9
# fits gemma4's 8192-token window with room for the prompt
# (code-heavy text runs ~3 chars/token)
MEMORY_TRANSCRIPT_MAX_CHARS = 16000
MEMORY_EXTRACT_LOG = MEMORY_DIR / ".extracted.json"
