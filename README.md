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
   any agent ───┘                DAEMON  ├─ Chroma (semantic search)
                                    ▲    └─ Lifecycle (dedup-lite today,
   session transcripts ─────────────┘       promote/decay in Plan 3)
   (SessionEnd hook → /memory/extract)
```

Two doors into the memory:

1. **Agents read & write directly** via four MCP tools (below).
2. **Auto-extraction:** when a session ends, its transcript runs through a
   five-stage pipeline — parse → **redact secrets** → distill with a local
   LLM → quality-gate (every memory must carry a verbatim evidence quote
   from the transcript; hallucinations die here; max 5 per session) →
   write as `proposed`, with full provenance. Near-duplicates reinforce the
   existing memory instead of piling up.

## Quick start

```bash
# 0. Requirements: Python 3.10+, Ollama running with the models pulled:
ollama pull nomic-embed-text   # embeddings
ollama pull gemma4             # extraction LLM

# 1. Install and start the daemon
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python server.py            # serves on 127.0.0.1:8899

# 2. Connect an agent — e.g. Claude Code:
claude mcp add --transport sse yaadein http://127.0.0.1:8899/sse
#    (Cursor and other MCP clients: add the same SSE URL in their MCP settings)

# 3. Optional: auto-extraction — add the SessionEnd hook below to
#    ~/.claude/settings.json so finished sessions are mined for memories.
```

## Tools (available to any connected MCP agent)

| Tool | What it does |
|---|---|
| `remember` | Save a durable fact (preference, decision, fact, gotcha) |
| `recall_memory` | Search memories, ranked; pass `project_path` for project scope |
| `memory_briefing` | Session-start digest: top facts, recent decisions, gotchas |
| `forget_memory` | Permanently delete a memory by id |

**Scopes:** memories are either user-wide (`"*"`) or bound to a project
(keyed by git remote URL, falling back to repo root path) — every repo
accumulates its own institutional memory, invisible to other projects.

**Storage:** SQLite (`memory_store/memories.db`) is the source of truth;
embeddings live in a dedicated Chroma collection. Every mutation is recorded
in an audit log. Inspect it anytime:

```bash
sqlite3 memory_store/memories.db "SELECT content, status, confidence FROM memories;"
```

## Automatic memory extraction

Add to `~/.claude/settings.json`:

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

Or trigger manually for any transcript:

```bash
curl -X POST http://127.0.0.1:8899/memory/extract \
  -H 'Content-Type: application/json' \
  -d '{"transcript_path": "~/.claude/projects/<project>/<session>.jsonl", "project_path": "/path/to/repo"}'
```

Each transcript is processed once (re-runs are no-ops until it changes).
Extraction failures are logged and retryable — a transcript is only marked
processed after a successful run.

## Development

```bash
./venv/bin/pytest            # full suite, no Ollama required
./venv/bin/pytest -m eval    # extraction-quality evals (needs live Ollama)
```

The extraction eval harness (`tests/test_extraction_eval.py`) is how the
distill prompt and quality gates get tuned — add fixture transcripts with
expected memories and measure.

## Roadmap (Plan 3)

- **Lifecycle engine:** contradiction chains (new facts supersede old, with
  human-in-the-loop for confirmed conflicts), promote/decay from usage.
- **Inspector CLI:** `yaad list | show | confirm | conflicts`.
- **Transcript watcher:** polling fallback for harnesses without hooks.

Design history lives in [docs/specs](docs/specs) and [docs/plans](docs/plans).
Yaadein began life inside [Recall](https://github.com/j-ai-01/rag-pipeline),
a local RAG engine, and was extracted once it grew its own identity.
