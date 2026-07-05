# Yaadein (यादें)

**One shared, persistent memory for all your AI agents — that fills itself.**

Every agent session starts as a goldfish: you explain your preferences to
Claude Code today, repeat them to Cursor tomorrow, and again next week when
the session resets. Yaadein is a local-first memory daemon that fixes that.
Any MCP-speaking agent connects to it and gets one brain: facts saved from
one agent are known to every other, and finished sessions are automatically
mined for durable memories. Fully local — SQLite + Chroma + Ollama, nothing
leaves your machine.

*Yaadein* is Hindi for "memories."

---

## How it works

```
   Claude Code ─┐                        ┌─ SQLite (source of truth:
   Cursor ──────┼── MCP (SSE) ──► YAADEIN │  scopes, provenance, audit log)
   any agent ───┘                DAEMON  ├─ Chroma (semantic search index)
                                    ▲    └─ Lifecycle (dedup-lite today,
   session transcripts ─────────────┘       promote/decay in Plan 3)
   (SessionEnd hook → /memory/extract)
```

Two doors into the memory:

1. **Agents read & write directly** via six MCP tools (see [Tools](#the-six-tools)).
2. **Auto-extraction:** when a session ends, its transcript runs through a
   five-stage pipeline — parse → **redact secrets** → distill with a local
   LLM → quality-gate → write as `proposed` with full provenance.

Since v2, every extraction pass also records an **episode** — a write-once
conversation record: a name-preserving summary (searchable by meaning), a
redacted verbatim excerpt (episodes never go blind, even if the transcript
is deleted), a transcript pointer, and links to the facts born in that pass.
**Facts for speed, episodes for story, transcript pointers for ground
truth** — and the briefing includes "recently discussed."

---

## Complete setup — every step

### Step 0: Prerequisites

| Requirement | Check | Install if missing |
|---|---|---|
| Python 3.10+ | `python3 --version` | `brew install python@3.11` |
| Ollama | `ollama --version` | https://ollama.com (or `brew install ollama`) |
| jq (for the hook) | `which jq` | `brew install jq` (preinstalled on recent macOS) |

Start Ollama and pull the two models Yaadein needs:

```bash
ollama serve &                  # skip if the Ollama app already runs
ollama pull nomic-embed-text    # embeddings (semantic search)
ollama pull gemma4              # extraction LLM (distills transcripts)
```

Verify Ollama is answering:

```bash
curl -s http://localhost:11434/api/tags | head -c 100   # should print JSON
```

### Step 1: Install Yaadein

```bash
cd ~/workplace/yaadein          # or wherever you cloned it
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Step 2: Start the daemon

```bash
./venv/bin/python server.py
```

You should see uvicorn report `Uvicorn running on http://127.0.0.1:8899`.
Leave it running (or background it: `./venv/bin/python server.py > yaadein.log 2>&1 &`).

Verify it's alive:

```bash
curl -s http://127.0.0.1:8899/health     # → {"status":"ok"}
```

> **After every reboot** the daemon must be started again. To make it
> automatic on macOS, create a `launchd` agent — or just alias it:
> `alias yaadein-up='cd ~/workplace/yaadein && ./venv/bin/python server.py > yaadein.log 2>&1 &'`

### Step 3: Connect your agents (one-time)

**Claude Code** — register at *user scope* so memory follows you into every
project (the default scope binds it to only the current directory — not what
you want):

```bash
claude mcp add --scope user --transport sse yaadein http://127.0.0.1:8899/sse
claude mcp list      # → yaadein: ... ✓ Connected
```

**Cursor / other MCP clients:** add an SSE MCP server with URL
`http://127.0.0.1:8899/sse` in their MCP settings.

**Kiro:** add to `~/.kiro/settings/mcp.json` (the `mcp-remote` bridge covers
Kiro builds that only speak stdio servers; needs node/npx):

```json
{
  "mcpServers": {
    "yaadein": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8899/sse"],
      "disabled": false,
      "autoApprove": ["recall_memory", "memory_briefing"]
    }
  }
}
```

Kiro then shares the same brain (all six tools). Auto-*mining* of Kiro
sessions additionally needs a `kiro-sessions` transcript parser — see
[Configuration](#configuration).

> ⚠️ **Tools load at session start.** A session that was already open when
> you registered the server will NOT have the memory tools — open a new one.

### Step 4: Enable auto-extraction (one-time)

Add the SessionEnd hook to `~/.claude/settings.json`. If the file already
has content, merge — don't overwrite. The result must contain:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "jq -c '{transcript_path, session_id, project_path: .cwd}' | curl -s -m 5 -X POST http://127.0.0.1:8899/memory/extract -H 'Content-Type: application/json' -d @- >/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

The `|| true` and `-m 5` matter: if the daemon is down, your session still
closes instantly and nothing breaks — that transcript is simply skipped
(you can mine it later, see [Manual extraction](#manual-extraction)).

### Step 5: Verify the whole loop

1. Open a **new** Claude Code session anywhere and say:
   *"remember that I prefer detailed commit messages"*
   → the agent should call the `remember` tool.
2. Check it landed:
   ```bash
   sqlite3 ~/workplace/yaadein/memory_store/memories.db \
     "SELECT content, status FROM memories;"
   ```
3. Open a session in a *different* project and ask:
   *"what do you know about my preferences?"*
   → the agent should call `recall_memory` and know it. One brain. ✓

> **`memory_store/` doesn't exist?** That's normal on a fresh install — the
> store is created lazily when the first memory is written. Empty brain,
> no folder.

---

## Using it day to day

### The six tools

| Tool | Who typically triggers it | What it does |
|---|---|---|
| `remember` | you ("remember this") or the agent when you state a clear preference/decision | Save a durable fact — lands `confirmed` |
| `recall_memory` | the agent, whenever your question touches preferences or project knowledge | Semantic search, ranked; `project_path` adds project scope |
| `memory_briefing` | agent at session start | Digest: top facts, recent decisions, gotchas, conflicts, recently discussed |
| `forget_memory` | you, essentially always | Permanent delete (audit-logged) |
| `recall_conversations` | the agent, when you refer to a past discussion | Search episode summaries by meaning, recency-weighted |
| `read_conversation` | the agent, after recall_conversations | One episode's summary + verbatim (redacted) excerpt + linked facts |

**Scopes:** memories are user-wide (`"*"`) or bound to a project — keyed by
git remote URL (fallback: repo root path), so the same repo is recognized
from any checkout. Recall always returns your user-wide memories plus the
current project's; other projects' memories stay invisible.

### What auto-extraction keeps (and rejects)

Kept: preferences, decisions **with reasons**, environment facts, gotchas.
Rejected: session-local trivia, anything derivable from the code, vague
observations, anything whose evidence quote isn't literally in the
transcript (hallucination guard), anything past the 5-per-session budget.
Everything extracted lands as **`proposed`** (labeled unconfirmed) with the
verbatim evidence quote and source session id attached.

**Secrets:** transcripts are scrubbed (AWS keys, GitHub/OpenAI-style tokens,
bearer tokens, private-key blocks, `password=`/`api_key=`-style assignments,
high-entropy strings) *before* the LLM ever sees the text.

### Near-real-time extraction (the watcher)

You don't have to wait for a session to end. Every 30 seconds
(`MEMORY_WATCH_INTERVAL_SECONDS` in config.py; 0 disables, raise it to
lighten the load on your machine) the daemon sweeps `~/.claude/projects/`
for transcripts modified within the last minute and re-mines them. This is safe by construction: unchanged transcripts are
skipped by content hash, and facts the LLM re-derives *reinforce* the
existing memory instead of duplicating it. So during a long working session
your memories stay at most ~5 minutes behind the conversation — and the
SessionEnd hook still provides the immediate final pass at close.

Deliberately **not** per-message: mid-thought extraction memorizes ideas you
were about to discard, and keeps the LLM running constantly. Minutes-fresh
is the sweet spot; end-of-session is the safety net.

### Manual extraction

Mine any transcript on demand (useful for sessions that ended while the
daemon was down, or other harnesses without hooks):

```bash
curl -X POST http://127.0.0.1:8899/memory/extract \
  -H 'Content-Type: application/json' \
  -d '{"transcript_path": "~/.claude/projects/<project-dir>/<session-id>.jsonl",
       "project_path": "/path/to/the/repo",
       "session_id": "<session-id>"}'
```

Each transcript is processed once per content-hash — re-POSTing is a no-op
until the file changes. Failed extractions (Ollama down, unparseable LLM
output) are **not** marked processed and can simply be retried.

### Inspecting the brain

**CLI** (the database is a plain file — `memory_store/memories.db`):

```bash
# everything it remembers
sqlite3 memory_store/memories.db "SELECT content, category, status, confidence FROM memories;"

# with receipts: evidence quote + which session taught it
sqlite3 memory_store/memories.db "SELECT content, evidence, source_session FROM memories;"

# what's still unconfirmed (auto-extracted, awaiting promotion)
sqlite3 memory_store/memories.db "SELECT content FROM memories WHERE status='proposed';"

# which memories actually get used
sqlite3 memory_store/memories.db "SELECT content, times_retrieved FROM memories ORDER BY times_retrieved DESC;"

# the full event history (add / retrieve / reinforce / forget)
sqlite3 memory_store/memories.db "SELECT ts, action, memory_id FROM audit_log ORDER BY id;"
```

**GUI:** `brew install --cask db-browser-for-sqlite`, then:

```bash
open -a "DB Browser for SQLite" ~/workplace/yaadein/memory_store/memories.db
```

Browse Data tab → `memories` table. Treat it as a **viewer**: pending edits
hold a write lock that can collide with the daemon ("database is locked").
Prefer `forget_memory` over hand-deleting rows so the audit log stays true.

### Wiping the memory

```bash
# stop the daemon first, then:
rm -rf ~/workplace/yaadein/memory_store
# restart the daemon — it starts with an empty brain
```

SQLite is the source of truth and Chroma is a derived index, so deleting
only `memory_store/chroma_db/` loses nothing but the search index.

---

## Configuration

Everything tunable lives in [config.py](config.py), and every knob can be
overridden per-run with a `YAADEIN_*` environment variable — no file edits:

| Env var | Default | What it controls |
|---|---|---|
| `YAADEIN_HOST` / `YAADEIN_PORT` | `127.0.0.1` / `8899` | where the daemon listens |
| `YAADEIN_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `YAADEIN_LLM_MODEL` / `YAADEIN_EMBED_MODEL` | `gemma4` / `nomic-embed-text` | extraction LLM / embedder |
| `YAADEIN_DATA_DIR` | `./memory_store` | where the brain lives |
| `YAADEIN_WATCH_INTERVAL` | `30` | watcher sweep seconds (0 = off) |
| `YAADEIN_WATCH_SOURCES` | Claude Code + Codex + Kiro | JSON list of watch sources (below) |
| `YAADEIN_TOP_K`, `YAADEIN_MAX_PER_SESSION`, `YAADEIN_CONFIDENCE_FLOOR`, `YAADEIN_REINFORCE_THRESHOLD`, `YAADEIN_TRANSCRIPT_MAX_CHARS` | see config.py | recall & gate tuning |
| `YAADEIN_EPISODE_EXCERPT_MAX_CHARS` | `6000` | verbatim excerpt cap per episode |
| `YAADEIN_EPISODE_RECENCY_WEIGHT` / `YAADEIN_EPISODE_RECENCY_HALFLIFE_DAYS` | `0.15` / `7.0` | recency bonus in conversation search |

**Watch sources** make harness support pluggable. Each source names a
transcript directory, a glob, a harness label, and a transcript `format`:

```json
[{"root": "~/.claude/projects", "glob": "*/*.jsonl",
  "harness": "claude-code", "format": "claude-jsonl"},
 {"root": "~/.codex/sessions", "glob": "*/*/*/*.jsonl",
  "harness": "codex", "format": "codex-jsonl"}]
```

Formats map to parsers in `yaadein/transcript.py` (`PARSERS`). A source
whose format has no parser yet is skipped with a warning at startup — the
Codex source reads Codex Desktop JSONL sessions, while the Kiro source ships
pre-configured and lights up automatically the day a `kiro-sessions` parser
is registered. **Adding a new harness = one parser function + one registry
line + one source entry.**

## Architecture in one paragraph

Each memory is a SQLite row (content, category, scope, `status`
proposed→confirmed→archived, confidence, evidence, source session,
retrieval counters, supersedence pointers) plus a Chroma entry under the
same id holding only the content's embedding. Reads that need *meaning*
("what's relevant to X?") ask Chroma for nearby ids, then SQLite for the
truth about each id (trust, scope, lifecycle) — Chroma nominates, SQLite
decides. Reads that need *reporting* (`memory_briefing`) never touch Chroma
at all. Writes hit SQLite first, then Chroma, and roll back the row if
indexing fails — a memory is never stored-but-unfindable. Every mutation is
audit-logged.

## Code map & how a memory flows

Every module now carries purpose docstrings — this table is the index:

| File | What it is | Depends on |
|---|---|---|
| `server.py` | Daemon entrypoint: FastAPI + MCP-over-SSE, `/memory/extract`, watcher startup | everything below |
| `config.py` | All knobs, env-overridable (`YAADEIN_*`) | — |
| `yaadein/schema.py` | SQLite tables + versioned migrations | — |
| `yaadein/types.py` | `Memory` and `Candidate` dataclasses | — |
| `yaadein/store.py` | **Truth layer**: CRUD on memories, every mutation audit-logged | schema, types |
| `yaadein/vector_index.py` | **Meaning layer**: Chroma embeddings, `Embedder` protocol | utils/chroma_client |
| `yaadein/scopes.py` | Project identity: git remote → repo root → path | — |
| `yaadein/service.py` | The brain's API: remember/propose/recall/briefing/forget/find_similar/reinforce | store, vector_index, scopes |
| `yaadein/mcp_tools.py` | The six MCP tool definitions + dispatch (descriptions steer agent behavior) | service, scopes |
| `yaadein/transcript.py` | Claude Code JSONL parser + `PARSERS` registry + tail truncation | — |
| `yaadein/redact.py` | Secret scrubbing (patterns + entropy) — runs before any LLM sees text | — |
| `yaadein/gates.py` | Hallucination defense: grounding, budget, confidence floor, batch dedupe | types, config |
| `yaadein/llm.py` | `TextGenerator` protocol + `OllamaGenerator` | config |
| `yaadein/extractor.py` | Pipeline orchestrator: hash idempotency + turn bookmark + reinforce-vs-propose | all of the above |
| `yaadein/watcher.py` | Finds recently-active transcripts, sniffs their project cwd | — |
| `utils/*` | Small helpers: chroma client, file hash, processed-log, ollama check | — |

**Flow 1 — explicit save** (agent calls `remember`):

```
MCP client ──SSE──► server.handle_call_tool
  └► mcp_tools.handle_memory_tool("remember")
       └► service.remember(content, ...)          status=confirmed
            ├► store.add(row)  ── SQLite INSERT + audit "add"
            └► vector_index.add(id, content)  ── embed via Ollama → Chroma
                 (on failure: store.forget(id) rolls the row back)
```

**Flow 2 — recall** (agent asks "what do we know about X?"):

```
service.recall(query, project_key)
  1. vector_index.query(query)      Chroma nominates ids by meaning
  2. store.get(id) per hit          SQLite decides: scope? archived? superseded?
  3. + keyword bonus, sort, top-5
  4. store.record_retrieval(ids)    counters tick → future promote/decay fuel
```

**Flow 3 — auto-extraction** (hook, watcher, or manual POST):

```
POST /memory/extract ──background──► extractor.extract(path)
  0. hash check: unchanged file? → skip.  bookmark: slice turns[seen:]
  1. transcript.parse_transcript    jsonl → clean Turns (new ones only)
  2. redact.redact                  secrets never reach the LLM
  3. llm.generate(distill prompt)   gemma proposes candidate facts
  4. llm.generate(summary prompt)   summary (LLM call #2), skipped if no episode index
  5. gates.apply_gates              no verbatim evidence → rejected; max 5
  6. per survivor: service.find_similar ≥ 0.85?
       yes → service.reinforce      (same fact re-learned = confidence +0.1)
       no  → service.propose        status=proposed, stamped with episode_id
  7. service.record_episode         redacted excerpt + transcript pointer,
                                    preset episode id, using the step-4 summary
  8. mark processed (hash + bookmark advanced)
     ordering is atomic-retryable (R9.1): any failure before step 8 leaves
     the transcript unprocessed — a retry re-runs the window; dedup bounds it
```

**Flow 4 — the watcher loop** (`server.start_transcript_watcher`):

```
every WATCH_INTERVAL seconds, per configured source (claude-code, codex, kiro…):
  format has a parser? ── no → skipped with a warning at startup
  watcher.find_recent_transcripts(root, glob)   modified in last 2×interval
  watcher.sniff_project_path(transcript)        cwd from the file's own entries
  └► Flow 3 for each — cheap when nothing changed (hash short-circuits)
```

## Development

```bash
./venv/bin/pytest            # full suite (~85 tests), no Ollama required
./venv/bin/pytest -m eval    # extraction-quality evals (needs live Ollama)
```

The eval harness (`tests/test_extraction_eval.py`) is how the distill
prompt and quality gates get tuned: add fixture transcripts with expected
memories and measure.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `claude mcp list` shows ✗ for yaadein | Daemon not running → `./venv/bin/python server.py`; then `curl :8899/health` |
| Agent session has no memory tools | Session predates registration → open a new session |
| Memories only exist in one project | Server registered at local scope → re-add with `--scope user` |
| `Address already in use` on start | A stale daemon holds the port → `lsof -nP -iTCP:8899 -sTCP:LISTEN`, kill that PID (use `-sTCP:LISTEN` — plain `lsof -ti :8899` also lists *client* connections, e.g. your browser) |
| `database is locked` | DB Browser has unsaved edits → Write Changes or close it |
| `memory_store/` missing | Nothing remembered yet — created on first write |
| Extraction produced nothing | Check daemon log; likely gates rejected everything (working as designed) or Ollama was down (retryable — re-POST) |
| Recall feels slow | The time is Ollama's embedding call, not the databases |

## Roadmap (Plan 3)

- **Lifecycle engine:** contradiction chains (new facts supersede old, with
  human-in-the-loop for confirmed conflicts), promote/decay from usage.
- **Inspector CLI:** `yaad list | show | confirm | conflicts`.
- **Full BM25 in recall** — deliberately deferred (YAGNI): today's recall is
  semantic search + a keyword bonus, which is plenty at one-sentence-fact
  scale. Upgrade trigger: exact-keyword memories ranking below fuzzy
  matches, or the store passing ~1,000 memories. One-function change
  (`recall` in `yaadein/service.py`).

Design history lives in [docs/specs](docs/specs) and [docs/plans](docs/plans).
Yaadein began life inside [Recall](https://github.com/j-ai-01/rag-pipeline),
a local RAG engine, and was extracted once it grew its own identity.
