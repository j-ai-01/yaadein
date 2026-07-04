# Agent Memory Extractor Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-extract durable memories from agent session transcripts — a Claude Code SessionEnd hook (or manual call) POSTs a transcript path to Recall, which parses, redacts, LLM-distills, quality-gates, and writes surviving facts as `proposed` memories with provenance.

**Architecture:** A five-stage pipeline (`parse → redact → distill → gate → write`) composed of small pure modules (`memory/transcript.py`, `memory/redact.py`, `memory/gates.py`) orchestrated by `memory/extractor.py`. The LLM sits behind a `TextGenerator` protocol (`memory/llm.py`) so every pipeline test runs with a fake; only the `@pytest.mark.eval` quality tests touch Ollama. Extraction is triggered by a new `POST /memory/extract` endpoint on the existing FastAPI server, runs as a background task, and is idempotent per transcript hash (reusing the `utils/ingest_tracker` pattern). Dedup-lite at write time: a candidate very similar to an existing same-scope memory reinforces it instead of duplicating; full lifecycle (LLM-judge, contradictions, promote/decay) is Plan 3.

**Tech Stack:** Python 3.10+, existing deps only (llama_index Ollama LLM, chromadb, FastAPI), stdlib `re`/`json`/`hashlib`.

**Branch context:** Builds on `feature/agent-memory-core` (Plan 1, unmerged) — `MemoryService`, `MemoryStore`, `MemoryVectorIndex`, `resolve_project_key`, and the MCP tools already exist. Continue committing on that branch.

## Global Constraints

- Fully local: no network calls except Ollama at `OLLAMA_BASE_URL`.
- No new pip dependencies.
- All new constants go in `config.py`.
- Category values exactly: `preference | decision | fact | gotcha`. Extracted memories land with status `proposed`. Explicit `remember` stays `confirmed`.
- Redaction runs BEFORE any LLM call — the generator must never see raw credentials.
- Grounding gate: a candidate whose `evidence_quote` is not literally in the (redacted) transcript is rejected.
- Budget: max `MEMORY_MAX_PER_SESSION = 5` memories per transcript; confidence floor `MEMORY_CONFIDENCE_FLOOR = 0.6`; reinforce threshold `MEMORY_REINFORCE_THRESHOLD = 0.9`.
- Extraction failures are logged and retryable; a transcript is marked processed only after a successful run. Never corrupts existing memories.
- Tests never require Ollama except tests marked `@pytest.mark.eval` (deselected by default via pytest.ini `addopts = -m "not eval"`).
- Every store mutation is audited (existing rule; `reinforce` must audit too).
- Follow existing test style: plain pytest functions, tmp_path fixtures, one behavior per test.

---

### Task 1: Config constants, Candidate type, and TextGenerator protocol

**Files:**
- Modify: `config.py` (append to the existing "Memory layer" block)
- Modify: `memory/types.py` (add `Candidate` dataclass after `Memory`)
- Create: `memory/llm.py`
- Test: `tests/test_memory_llm.py`

**Interfaces:**
- Consumes: `config.LLM_MODEL`, `config.OLLAMA_BASE_URL` (existing).
- Produces:
  - `config`: `MEMORY_MAX_PER_SESSION = 5`, `MEMORY_CONFIDENCE_FLOOR = 0.6`, `MEMORY_REINFORCE_THRESHOLD = 0.9`, `MEMORY_TRANSCRIPT_MAX_CHARS = 24000`, `MEMORY_EXTRACT_LOG = MEMORY_DIR / ".extracted.json"`.
  - `memory.types.Candidate` — dataclass: `content: str`, `category: str`, `scope: str` ("user" | "project"), `confidence: float`, `evidence_quote: str`.
  - `memory.llm.TextGenerator` — Protocol with `generate(prompt: str) -> str`.
  - `memory.llm.OllamaGenerator` — implements it via `llama_index.llms.ollama.Ollama` (lazy imports inside `__init__`, like `OllamaEmbedder`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_llm.py
from memory.llm import TextGenerator
from memory.types import Candidate


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.llm'`

- [ ] **Step 3: Write minimal implementation**

Append to the memory block in `config.py`:

```python
MEMORY_MAX_PER_SESSION = 5
MEMORY_CONFIDENCE_FLOOR = 0.6
MEMORY_REINFORCE_THRESHOLD = 0.9
MEMORY_TRANSCRIPT_MAX_CHARS = 24000
MEMORY_EXTRACT_LOG = MEMORY_DIR / ".extracted.json"
```

Append to `memory/types.py`:

```python
@dataclass
class Candidate:
    content: str
    category: str
    scope: str  # "user" | "project"
    confidence: float
    evidence_quote: str
```

Create `memory/llm.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_llm.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add config.py memory/types.py memory/llm.py tests/test_memory_llm.py
git commit -m "feat(memory): extraction config, Candidate type, TextGenerator protocol"
```

---

### Task 2: Transcript parser

**Files:**
- Create: `memory/transcript.py`
- Test: `tests/test_memory_transcript.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure module).
- Produces:
  - `memory.transcript.Turn` — dataclass: `role: str` ("user" | "assistant"), `text: str`.
  - `memory.transcript.parse_transcript(path: Path) -> List[Turn]` — tolerant Claude Code JSONL reader.
  - `memory.transcript.transcript_text(turns: List[Turn], max_chars: int) -> str` — `"USER: ..."` / `"ASSISTANT: ..."` lines; keeps the TAIL (most recent) when over budget, cutting at a line boundary.

**Claude Code JSONL facts (verified against real transcripts):** each line is JSON. Message lines have `type: "user"` or `"assistant"` and `message: {role, content}`. `content` is either a string (typed user prompt) or a list of blocks: `{type: "text", text}`, `{type: "tool_use", name, input}`, `{type: "tool_result", ...}`, `{type: "thinking", ...}`. Non-message lines (`attachment`, `system`, `file-history-snapshot`, `last-prompt`, `ai-title`, `permission-mode`, ...) must be skipped, as must malformed lines, `tool_result`/`thinking` blocks, harness-injected user strings starting with `<`, and empty turns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_transcript.py
import json
from memory.transcript import Turn, parse_transcript, transcript_text


def write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries))


def user_str(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def assistant_blocks(blocks):
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}}


def test_parses_user_strings_and_assistant_text(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        {"type": "file-history-snapshot"},
        user_str("I prefer pytest over unittest"),
        assistant_blocks([
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Noted, pytest it is."},
        ]),
    ])
    turns = parse_transcript(p)
    assert turns == [
        Turn("user", "I prefer pytest over unittest"),
        Turn("assistant", "Noted, pytest it is."),
    ]


def test_tool_use_summarized_and_tool_results_skipped(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        assistant_blocks([
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "file1\nfile2"},
        ]}},
    ])
    turns = parse_transcript(p)
    assert turns == [Turn("assistant", "Checking. [tool: Bash]")]


def test_skips_malformed_lines_and_harness_injected_user_text(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "not json at all\n"
        + json.dumps(user_str("<system-reminder>ignore</system-reminder>")) + "\n"
        + json.dumps(user_str("real question"))
    )
    assert parse_transcript(p) == [Turn("user", "real question")]


def test_transcript_text_formats_roles():
    turns = [Turn("user", "hi"), Turn("assistant", "hello")]
    assert transcript_text(turns, max_chars=1000) == "USER: hi\nASSISTANT: hello"


def test_transcript_text_keeps_tail_when_over_budget():
    turns = [Turn("user", "old " * 50), Turn("assistant", "recent answer")]
    out = transcript_text(turns, max_chars=40)
    assert "recent answer" in out
    assert "old" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_transcript.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.transcript'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/transcript.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


def _text_from_blocks(blocks, include_tools: bool) -> str:
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
        elif block.get("type") == "tool_use" and include_tools:
            parts.append(f"[tool: {block.get('name', 'unknown')}]")
    return " ".join(parts)


def parse_transcript(path: Path) -> List[Turn]:
    turns: List[Turn] = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict) or entry.get("type") not in ("user", "assistant"):
            continue
        message = entry.get("message") or {}
        content = message.get("content")
        role = entry["type"]
        if isinstance(content, str):
            text = content.strip()
            if role == "user" and text.startswith("<"):
                continue  # harness-injected reminders/commands, not user speech
        elif isinstance(content, list):
            text = _text_from_blocks(content, include_tools=(role == "assistant"))
        else:
            continue
        if text:
            turns.append(Turn(role, text))
    return turns


def transcript_text(turns: List[Turn], max_chars: int) -> str:
    lines = [f"{turn.role.upper()}: {turn.text}" for turn in turns]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    return tail[newline + 1:] if newline != -1 else tail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_transcript.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add memory/transcript.py tests/test_memory_transcript.py
git commit -m "feat(memory): tolerant Claude Code transcript parser"
```

---

### Task 3: Secrets redaction

**Files:**
- Create: `memory/redact.py`
- Test: `tests/test_memory_redact.py`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces: `memory.redact.redact(text: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_redact.py
from memory.redact import redact


def test_redacts_aws_access_key():
    out = redact("creds: AKIAIOSFODNN7EXAMPLE done")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "done" in out


def test_redacts_key_value_assignments():
    out = redact("api_key = sk_live_abc123 and password: hunter2")
    assert "sk_live_abc123" not in out
    assert "hunter2" not in out


def test_redacts_bearer_tokens():
    out = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out


def test_redacts_private_key_blocks():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKC\n-----END RSA PRIVATE KEY-----"
    out = redact(f"here {block} there")
    assert "MIIEowIBAAKC" not in out
    assert "there" in out


def test_redacts_github_and_openai_style_tokens():
    out = redact("use ghp_16C7e42F292c6912E7710c838347Ae178B4a and sk-proj-abcdefghij1234567890")
    assert "ghp_16C7e42F292c6912E7710c838347Ae178B4a" not in out
    assert "sk-proj-abcdefghij1234567890" not in out


def test_redacts_long_high_entropy_tokens():
    out = redact("token was g9X2kQ7vZp4mW8rT1nB5cY3hL6jD0aFs")
    assert "g9X2kQ7vZp4mW8rT1nB5cY3hL6jD0aFs" not in out


def test_leaves_normal_prose_and_paths_alone():
    text = "The user prefers pytest; config lives in /Users/jai/workplace/rag-pipeline/config.py"
    assert redact(text) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_redact.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.redact'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/redact.py`:

```python
import math
import re

_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\b(\s*[:=]\s*)\S+"),
     r"\1\2[REDACTED]"),
]

_CANDIDATE_TOKEN = re.compile(r"\S{28,}")


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def _looks_like_secret(token: str) -> bool:
    if "/" in token or "\\" in token:
        return False  # paths and URLs
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_digit and has_alpha and _entropy(token) > 4.0


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return _CANDIDATE_TOKEN.sub(
        lambda m: "[REDACTED_HIGH_ENTROPY]" if _looks_like_secret(m.group()) else m.group(),
        text,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_redact.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add memory/redact.py tests/test_memory_redact.py
git commit -m "feat(memory): pattern + entropy secrets redaction for transcripts"
```

---

### Task 4: Quality gates

**Files:**
- Create: `memory/gates.py`
- Test: `tests/test_memory_gates.py`

**Interfaces:**
- Consumes: `memory.types.Candidate` (Task 1), `config.MEMORY_MAX_PER_SESSION`, `config.MEMORY_CONFIDENCE_FLOOR`.
- Produces: `memory.gates.apply_gates(candidates: List[Candidate], transcript: str) -> List[Candidate]` — enforces, in order: valid category; scope in ("user", "project"); confidence ≥ floor; content length 10–300 chars; grounding (normalized `evidence_quote` is a substring of the normalized transcript); in-batch dedupe on normalized content; sort by confidence desc; cap at max per session.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_gates.py
from memory.gates import apply_gates
from memory.types import Candidate

TRANSCRIPT = "USER: I prefer pytest over unittest because less boilerplate\nASSISTANT: Noted."


def cand(**overrides):
    base = dict(
        content="User prefers pytest over unittest",
        category="preference", scope="user", confidence=0.9,
        evidence_quote="I prefer pytest over unittest",
    )
    base.update(overrides)
    return Candidate(**base)


def test_grounded_valid_candidate_survives():
    assert len(apply_gates([cand()], TRANSCRIPT)) == 1


def test_hallucinated_evidence_rejected():
    c = cand(evidence_quote="I love unittest dearly")
    assert apply_gates([c], TRANSCRIPT) == []


def test_grounding_is_whitespace_and_case_insensitive():
    c = cand(evidence_quote="i PREFER pytest   over unittest")
    assert len(apply_gates([c], TRANSCRIPT)) == 1


def test_invalid_category_rejected():
    assert apply_gates([cand(category="opinion")], TRANSCRIPT) == []


def test_low_confidence_rejected():
    assert apply_gates([cand(confidence=0.3)], TRANSCRIPT) == []


def test_too_short_content_rejected():
    assert apply_gates([cand(content="pytest")], TRANSCRIPT) == []


def test_batch_deduped_on_normalized_content():
    dupes = [cand(), cand(content="user prefers PYTEST over unittest")]
    assert len(apply_gates(dupes, TRANSCRIPT)) == 1


def test_budget_keeps_highest_confidence():
    many = [
        cand(content=f"Distinct durable fact number {i} about pytest", confidence=0.6 + i * 0.05,
             evidence_quote="I prefer pytest over unittest")
        for i in range(8)
    ]
    kept = apply_gates(many, TRANSCRIPT)
    assert len(kept) == 5
    assert kept[0].confidence >= kept[-1].confidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.gates'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/gates.py`:

```python
from typing import List

from config import MEMORY_CONFIDENCE_FLOOR, MEMORY_MAX_PER_SESSION
from memory.types import Candidate

VALID_CATEGORIES = {"preference", "decision", "fact", "gotcha"}
VALID_SCOPES = {"user", "project"}
_MIN_CONTENT_CHARS = 10
_MAX_CONTENT_CHARS = 300


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _passes(candidate: Candidate, normalized_transcript: str) -> bool:
    if candidate.category not in VALID_CATEGORIES:
        return False
    if candidate.scope not in VALID_SCOPES:
        return False
    if candidate.confidence < MEMORY_CONFIDENCE_FLOOR:
        return False
    if not (_MIN_CONTENT_CHARS <= len(candidate.content) <= _MAX_CONTENT_CHARS):
        return False
    if _normalize(candidate.evidence_quote) not in normalized_transcript:
        return False  # hallucinated evidence
    return True


def apply_gates(candidates: List[Candidate], transcript: str) -> List[Candidate]:
    normalized_transcript = _normalize(transcript)
    survivors, seen = [], set()
    for candidate in candidates:
        if not _passes(candidate, normalized_transcript):
            continue
        key = _normalize(candidate.content)
        if key in seen:
            continue
        seen.add(key)
        survivors.append(candidate)
    survivors.sort(key=lambda c: c.confidence, reverse=True)
    return survivors[:MEMORY_MAX_PER_SESSION]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_gates.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add memory/gates.py tests/test_memory_gates.py
git commit -m "feat(memory): extraction quality gates with grounding check"
```

---

### Task 5: Store.reinforce + Service.propose/find_similar/reinforce

**Files:**
- Modify: `memory/store.py` (add `reinforce` method after `set_status`)
- Modify: `memory/service.py` (add `propose`, `find_similar`, `reinforce` methods after `remember`)
- Test: `tests/test_memory_propose_reinforce.py`

**Interfaces:**
- Consumes: existing `MemoryStore`, `MemoryService`, `MemoryVectorIndex`, `Memory`.
- Produces:
  - `MemoryStore.reinforce(memory_id: str, source_session: Optional[str] = None) -> None` — `confidence = MIN(1.0, confidence + 0.1)`, audits `("reinforce", memory_id, source_session)`.
  - `MemoryService.propose(content: str, category: str, scope_type: str, scope_key: str, confidence: float, evidence: Optional[str] = None, source_harness: Optional[str] = None, source_session: Optional[str] = None) -> Memory` — status `proposed`; writes store row AND vector index (rolling back the row if indexing fails, same pattern as `remember`).
  - `MemoryService.find_similar(content: str, scope_type: str, scope_key: str) -> Optional[Tuple[Memory, float]]` — top vector hits (top_k=5), first non-archived, non-superseded memory in the SAME scope, with its similarity.
  - `MemoryService.reinforce(memory_id: str, source_session: Optional[str] = None) -> None` — delegates to the store.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_propose_reinforce.py
import math
import pytest
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class ExplodingEmbedder:
    def embed(self, text):
        raise RuntimeError("ollama down")


def make_service(tmp_path, embedder=None):
    store = MemoryStore(tmp_path / "memories.db")
    index = MemoryVectorIndex(
        chroma_dir=tmp_path / "chroma",
        embedder=embedder or FakeEmbedder(),
        collection_name="test_memories",
    )
    return MemoryService(store=store, vector_index=index), store


def test_propose_lands_proposed_with_provenance(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.propose(
        content="User prefers pytest for testing",
        category="preference", scope_type="user", scope_key="*",
        confidence=0.8, evidence="I prefer pytest",
        source_harness="claude-code", source_session="sess-1",
    )
    row = store.get(saved.id)
    assert row.status == "proposed"
    assert row.confidence == 0.8
    assert row.evidence == "I prefer pytest"
    assert row.source_session == "sess-1"


def test_propose_rolls_back_on_embed_failure(tmp_path):
    service, store = make_service(tmp_path, embedder=ExplodingEmbedder())
    with pytest.raises(RuntimeError):
        service.propose(
            content="User prefers pytest for testing",
            category="preference", scope_type="user", scope_key="*",
            confidence=0.8,
        )
    assert store.list() == []


def test_find_similar_matches_same_scope_only(tmp_path):
    service, _ = make_service(tmp_path)
    service.propose(
        content="Deploys use the blue pipeline", category="fact",
        scope_type="project", scope_key="repo-a", confidence=0.9,
    )
    hit = service.find_similar("deploy pipeline colour", "project", "repo-a")
    assert hit is not None and hit[0].content == "Deploys use the blue pipeline"
    assert service.find_similar("deploy pipeline colour", "project", "repo-b") is None


def test_reinforce_bumps_confidence_capped_and_audited(tmp_path):
    service, store = make_service(tmp_path)
    saved = service.propose(
        content="User prefers pytest for testing", category="preference",
        scope_type="user", scope_key="*", confidence=0.95,
    )
    service.reinforce(saved.id, source_session="sess-2")
    assert store.get(saved.id).confidence == 1.0
    actions = [row["action"] for row in store.audit_entries()]
    assert "reinforce" in actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_propose_reinforce.py -v`
Expected: FAIL with `AttributeError: 'MemoryService' object has no attribute 'propose'`

- [ ] **Step 3: Write minimal implementation**

Add to `memory/store.py` after `set_status`:

```python
    def reinforce(self, memory_id: str, source_session: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE memories SET confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
            (memory_id,),
        )
        self._audit("reinforce", memory_id, source_session)
        self._conn.commit()
```

Add to `memory/service.py` after `remember` (adjust imports: `from typing import List, Optional, Tuple`):

```python
    def propose(
        self,
        content: str,
        category: str,
        scope_type: str,
        scope_key: str,
        confidence: float,
        evidence: Optional[str] = None,
        source_harness: Optional[str] = None,
        source_session: Optional[str] = None,
    ) -> Memory:
        memory = Memory(
            id="", content=content, category=category,
            scope_type=scope_type, scope_key=scope_key,
            status="proposed", confidence=confidence, evidence=evidence,
            source_harness=source_harness, source_session=source_session,
        )
        saved = self._store.add(memory)
        try:
            self._index.add(saved.id, saved.content)
        except Exception:
            self._store.forget(saved.id)
            raise
        return saved

    def find_similar(
        self, content: str, scope_type: str, scope_key: str
    ) -> Optional[Tuple[Memory, float]]:
        for memory_id, similarity in self._index.query(content, top_k=5):
            memory = self._store.get(memory_id)
            if memory is None or memory.status == "archived" or memory.superseded_by:
                continue
            if memory.scope_type == scope_type and memory.scope_key == scope_key:
                return memory, similarity
        return None

    def reinforce(self, memory_id: str, source_session: Optional[str] = None) -> None:
        self._store.reinforce(memory_id, source_session)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_propose_reinforce.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite**

Run: `./venv/bin/pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add memory/store.py memory/service.py tests/test_memory_propose_reinforce.py
git commit -m "feat(memory): propose/find_similar/reinforce for the extraction write path"
```

---

### Task 6: Extractor orchestration

**Files:**
- Create: `memory/extractor.py`
- Test: `tests/test_memory_extractor.py`

**Interfaces:**
- Consumes: `parse_transcript`/`transcript_text` (Task 2), `redact` (Task 3), `apply_gates` (Task 4), `Candidate` (Task 1), `MemoryService.propose/find_similar/reinforce` (Task 5), `resolve_project_key` (Plan 1), `utils.file_hash.file_hash(path: Path) -> str`, `utils.ingest_tracker.load_ingested/save_ingested`, config constants (Task 1).
- Produces:
  - `memory.extractor.ExtractionResult` — dataclass: `written: List[str]`, `reinforced: List[str]`, `skipped: int`, `already_processed: bool = False`, `error: Optional[str] = None`.
  - `memory.extractor.Extractor(service: MemoryService, generator: TextGenerator, extract_log: Path)` with `extract(transcript_path: Path, source_harness: str = "claude-code", project_path: Optional[str] = None, session_id: Optional[str] = None) -> ExtractionResult`.
  - `memory.extractor.build_extractor() -> Extractor` — factory wiring `get_memory_service()`, `OllamaGenerator()`, `config.MEMORY_EXTRACT_LOG` (used by the server; needs Ollama, not unit-tested).
  - `memory.extractor._parse_candidates(raw: str) -> List[Candidate]` (module-private but tested).

Behavior: skip (with `already_processed=True`) if the transcript hash is already in the extract log; parse → redact → distill (prompt below) → gate → for each survivor map scope (`"project"` only when `project_path` given, else user), then `find_similar`: similarity ≥ `MEMORY_REINFORCE_THRESHOLD` → `reinforce`, else `propose`. Mark the hash processed ONLY on success. Any exception from the generator or JSON parsing → return `ExtractionResult` with `error` set, nothing marked processed (retryable), no partial store corruption (per-candidate writes are individually atomic via Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_extractor.py
import json
import math
from memory.extractor import Extractor, _parse_candidates
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.vector_index import MemoryVectorIndex


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
    assert store.get(existing.id).confidence == 0.8


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
    assert _parse_candidates("no json here") == []
    assert _parse_candidates('[{"content": 42}]') == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_memory_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory.extractor'`

- [ ] **Step 3: Write minimal implementation**

Create `memory/extractor.py`:

```python
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import MEMORY_REINFORCE_THRESHOLD, MEMORY_TRANSCRIPT_MAX_CHARS
from memory.gates import apply_gates
from memory.llm import TextGenerator
from memory.redact import redact
from memory.scopes import USER_SCOPE_KEY, resolve_project_key
from memory.service import MemoryService
from memory.transcript import parse_transcript, transcript_text
from memory.types import Candidate
from utils.file_hash import file_hash
from utils.ingest_tracker import load_ingested, save_ingested

logger = logging.getLogger(__name__)

_DISTILL_PROMPT = """You are a memory extraction system. Read the conversation transcript \
below and extract durable facts worth remembering in future sessions.

Extract ONLY:
- preferences the user stated (tools, style, workflow)
- decisions made, with their reasons
- environment facts or project conventions not obvious from the code
- gotchas: surprising problems and their causes

Do NOT extract:
- session-local details (current bug, specific line numbers)
- anything derivable by reading the code itself
- vague observations about the user

Return a JSON array and nothing else. Each item:
{{"content": "<one distilled fact, a single sentence>", \
"category": "preference|decision|fact|gotcha", \
"scope": "user|project", \
"confidence": 0.0-1.0, \
"evidence_quote": "<short verbatim quote from the transcript proving this>"}}

Return [] if nothing qualifies.

TRANSCRIPT:
{transcript}

JSON:"""

_REQUIRED_KEYS = {"content", "category", "scope", "confidence", "evidence_quote"}


def _parse_candidates(raw: str) -> List[Candidate]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    candidates = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not _REQUIRED_KEYS <= set(item):
            continue
        try:
            candidates.append(Candidate(
                content=str(item["content"]),
                category=str(item["category"]),
                scope=str(item["scope"]),
                confidence=float(item["confidence"]),
                evidence_quote=str(item["evidence_quote"]),
            ))
        except (TypeError, ValueError):
            continue
    return candidates


@dataclass
class ExtractionResult:
    written: List[str] = field(default_factory=list)
    reinforced: List[str] = field(default_factory=list)
    skipped: int = 0
    already_processed: bool = False
    error: Optional[str] = None


class Extractor:
    def __init__(self, service: MemoryService, generator: TextGenerator, extract_log: Path):
        self._service = service
        self._generator = generator
        self._extract_log = extract_log

    def extract(
        self,
        transcript_path: Path,
        source_harness: str = "claude-code",
        project_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ExtractionResult:
        transcript_hash = file_hash(transcript_path)
        processed = load_ingested(self._extract_log)
        if processed.get(str(transcript_path)) == transcript_hash:
            return ExtractionResult(already_processed=True)

        turns = parse_transcript(transcript_path)
        text = transcript_text(turns, MEMORY_TRANSCRIPT_MAX_CHARS)
        if not text.strip():
            self._mark_processed(processed, transcript_path, transcript_hash)
            return ExtractionResult()

        clean = redact(text)
        try:
            raw = self._generator.generate(_DISTILL_PROMPT.format(transcript=clean))
        except Exception as e:
            logger.exception("distillation failed for %s", transcript_path)
            return ExtractionResult(error=str(e))

        candidates = _parse_candidates(raw)
        survivors = apply_gates(candidates, clean)
        project_key = resolve_project_key(project_path) if project_path else None

        result = ExtractionResult(skipped=len(candidates) - len(survivors))
        for candidate in survivors:
            if candidate.scope == "project" and project_key:
                scope_type, scope_key = "project", project_key
            else:
                scope_type, scope_key = "user", USER_SCOPE_KEY
            similar = self._service.find_similar(candidate.content, scope_type, scope_key)
            if similar and similar[1] >= MEMORY_REINFORCE_THRESHOLD:
                self._service.reinforce(similar[0].id, session_id)
                result.reinforced.append(similar[0].id)
            else:
                saved = self._service.propose(
                    content=candidate.content, category=candidate.category,
                    scope_type=scope_type, scope_key=scope_key,
                    confidence=candidate.confidence, evidence=candidate.evidence_quote,
                    source_harness=source_harness, source_session=session_id,
                )
                result.written.append(saved.id)

        self._mark_processed(processed, transcript_path, transcript_hash)
        return result

    def _mark_processed(self, processed: dict, path: Path, transcript_hash: str) -> None:
        processed[str(path)] = transcript_hash
        self._extract_log.parent.mkdir(parents=True, exist_ok=True)
        save_ingested(processed, self._extract_log)


def build_extractor() -> "Extractor":
    from config import MEMORY_EXTRACT_LOG
    from memory.llm import OllamaGenerator
    from memory.service import get_memory_service

    return Extractor(
        service=get_memory_service(),
        generator=OllamaGenerator(),
        extract_log=MEMORY_EXTRACT_LOG,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_memory_extractor.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full suite**

Run: `./venv/bin/pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add memory/extractor.py tests/test_memory_extractor.py
git commit -m "feat(memory): five-stage extraction pipeline with dedup-lite and idempotency"
```

---

### Task 7: POST /memory/extract endpoint

**Files:**
- Modify: `mcp_server.py` (add `ExtractRequest` model near `QueryRequest`; add `_run_extraction` + endpoint after the `/query/stream` endpoint; add `BackgroundTasks` to the fastapi import)
- Test: `tests/test_extract_endpoint.py`

**Interfaces:**
- Consumes: `memory.extractor.build_extractor` (Task 6, imported lazily inside `_run_extraction` — never at module import).
- Produces: `POST /memory/extract` accepting JSON `{transcript_path: str, project_path?: str, session_id?: str, harness?: str = "claude-code"}` — returns 404 `{"error": ...}` if the path doesn't exist, else 200 `{"status": "queued", "transcript": <path>}` and runs extraction as a FastAPI background task. `mcp_server._run_extraction(transcript_path, project_path, session_id, harness)` is module-level so tests can monkeypatch it; it catches and logs all exceptions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract_endpoint.py
from fastapi.testclient import TestClient

import mcp_server


def test_extract_queues_background_extraction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_server, "_run_extraction",
        lambda *args: calls.append(args),
    )
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}")
    client = TestClient(mcp_server.app)
    resp = client.post("/memory/extract", json={
        "transcript_path": str(transcript),
        "project_path": "/some/repo",
        "session_id": "sess-1",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    assert calls == [(str(transcript), "/some/repo", "sess-1", "claude-code")]


def test_extract_404_for_missing_transcript(tmp_path):
    client = TestClient(mcp_server.app)
    resp = client.post("/memory/extract", json={
        "transcript_path": str(tmp_path / "nope.jsonl"),
    })
    assert resp.status_code == 404
    assert "error" in resp.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_extract_endpoint.py -v`
Expected: FAIL with 404/405 (route not defined) or AttributeError on `_run_extraction`

- [ ] **Step 3: Write minimal implementation**

In `mcp_server.py`, change the fastapi import line to:

```python
from fastapi import BackgroundTasks, FastAPI, Request
```

Add after the `QueryRequest` model:

```python
class ExtractRequest(BaseModel):
    transcript_path: str
    project_path: Optional[str] = None
    session_id: Optional[str] = None
    harness: str = "claude-code"
```

Add after the `/query/stream` endpoint:

```python
def _run_extraction(
    transcript_path: str,
    project_path: Optional[str],
    session_id: Optional[str],
    harness: str,
) -> None:
    from memory.extractor import build_extractor

    try:
        result = build_extractor().extract(
            Path(transcript_path),
            source_harness=harness,
            project_path=project_path,
            session_id=session_id,
        )
        logging.getLogger(__name__).info(
            "extraction for %s: %d written, %d reinforced, error=%s",
            transcript_path, len(result.written), len(result.reinforced), result.error,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "memory extraction failed for %s", transcript_path
        )


@app.post("/memory/extract")
async def memory_extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    path = Path(req.transcript_path).expanduser()
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Transcript not found: {req.transcript_path}"},
        )
    background_tasks.add_task(
        _run_extraction, str(path), req.project_path, req.session_id, req.harness
    )
    return {"status": "queued", "transcript": str(path)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_extract_endpoint.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite**

Run: `./venv/bin/pytest -q`
Expected: all tests pass (test_ui_endpoints.py must not regress)

- [ ] **Step 6: Commit**

```bash
git add mcp_server.py tests/test_extract_endpoint.py
git commit -m "feat(memory): POST /memory/extract endpoint with background extraction"
```

---

### Task 8: Eval harness (pytest marker + fixture + fake and real eval tests)

**Files:**
- Modify: `pytest.ini` (add `markers` and `addopts`)
- Create: `tests/fixtures/transcripts/pytest_preference.jsonl`
- Create: `tests/test_extraction_eval.py`

**Interfaces:**
- Consumes: `Extractor`/`build_extractor` pieces (Task 6), `MemoryStore`/`MemoryVectorIndex`/`MemoryService` (Plan 1), `OllamaGenerator` (Task 1), `utils.ollama_check.check_ollama_running() -> bool` (existing).
- Produces: an `eval` pytest marker deselected by default (`pytest` runs stay Ollama-free; `pytest -m eval` runs quality evals against real gemma4 + nomic-embed-text). This is the harness for tuning the distill prompt and gates.

- [ ] **Step 1: Update pytest.ini**

Replace `pytest.ini` content with:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    eval: extraction-quality evals that need a local Ollama LLM (deselected by default)
addopts = -m "not eval"
```

- [ ] **Step 2: Create the fixture transcript**

Create `tests/fixtures/transcripts/pytest_preference.jsonl` (each line is one JSON object):

```
{"type": "user", "message": {"role": "user", "content": "Let's add tests to this repo. I prefer pytest over unittest - way less boilerplate."}}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Agreed, pytest it is. I'll set up the test directory."}]}}
{"type": "user", "message": {"role": "user", "content": "Also heads up: the staging deploy fails unless you export AWS_REGION=us-east-1 first. Bit us twice this week."}}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Good to know - that is a classic gotcha. I'll note it in the runbook."}]}}
{"type": "user", "message": {"role": "user", "content": "What time is it?"}}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "I cannot check the clock, but your terminal can: run date."}]}}
```

- [ ] **Step 3: Write the eval tests**

```python
# tests/test_extraction_eval.py
import math
from pathlib import Path

import pytest

from memory.extractor import Extractor
from memory.service import MemoryService
from memory.store import MemoryStore
from memory.vector_index import MemoryVectorIndex
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
    from memory.llm import OllamaGenerator
    from memory.vector_index import OllamaEmbedder

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
```

- [ ] **Step 4: Run to verify default deselection and the smoke test**

Run: `./venv/bin/pytest tests/test_extraction_eval.py -v`
Expected: 1 passed, 1 deselected

Run: `./venv/bin/pytest -q`
Expected: full suite passes, eval deselected

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/fixtures/transcripts/pytest_preference.jsonl tests/test_extraction_eval.py
git commit -m "feat(memory): extraction eval harness behind pytest eval marker"
```

---

### Task 9: README — automatic extraction + hook setup

**Files:**
- Modify: `README.md` (extend the "Cross-Agent Memory" section: replace its final paragraph — the one beginning "Auto-extraction of memories from session transcripts is the next phase" — with the content below)

**Interfaces:**
- Consumes: endpoint contract from Task 7.
- Produces: user-facing docs; no code.

- [ ] **Step 1: Replace the closing paragraph of the Cross-Agent Memory section with:**

```markdown
### Automatic memory extraction

Recall can mine finished agent sessions for durable facts. A transcript is
parsed, **scrubbed of secrets**, distilled by your local LLM into candidate
facts, quality-gated (hallucinated evidence is rejected, max 5 per session),
and written as `proposed` memories with provenance. Near-duplicates reinforce
the existing memory instead of piling up.

Trigger it from a Claude Code `SessionEnd` hook — add to
`~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "jq -c '{transcript_path, session_id, project_path: .cwd}' | curl -s -m 5 -X POST http://localhost:8765/memory/extract -H 'Content-Type: application/json' -d @- >/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

Or trigger manually for any transcript:

```bash
curl -X POST http://localhost:8765/memory/extract \
  -H 'Content-Type: application/json' \
  -d '{"transcript_path": "~/.claude/projects/<project>/<session>.jsonl", "project_path": "/path/to/repo"}'
```

Each transcript is processed once (re-runs are no-ops until it changes).
Extraction quality evals live behind `pytest -m eval` (requires Ollama).

Still to come (Plan 3): the lifecycle engine (contradiction handling,
promote/decay) and a memory inspector CLI — see
`docs/specs/2026-07-03-agent-memory-design.md`.
```

- [ ] **Step 2: Run the suite once (docs-only sanity)**

Run: `./venv/bin/pytest -q`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: automatic memory extraction and SessionEnd hook setup"
```

---

## Out of scope (Plan 3)

- Lifecycle engine: LLM-judge gray-zone dedup, contradiction chains + `conflict_with` flagging, promote/decay sweep.
- Inspector CLI.
- The polling transcript **watcher** (spec's fallback trigger for harnesses without hooks): the manual `POST /memory/extract` covers those harnesses today; a background poller can reuse the same `Extractor` + hash log unchanged.

The extractor's write path (`propose`/`reinforce`) and the schema already carry everything Plan 3 needs.
