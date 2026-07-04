# Agent Memory Layer for Recall — Design

**Date:** 2026-07-03
**Status:** Approved design, pre-implementation
**Working name:** Recall Memory (internal codename during design: Engram)

## One-liner

Evolve Recall into a local-first memory layer that gives all of a developer's
AI agents (Claude Code, Cursor, any MCP client) one shared, learning brain —
memories are auto-extracted from agent session transcripts, confirmed by use,
and retrieved by any connected agent.

## Problem

Every agent session starts as a goldfish. Developers re-explain preferences,
project conventions, and past decisions to Claude Code today, Cursor tomorrow,
and the same tool again next week. Session memory dies with the process;
nothing is shared across harnesses. Recall already solves local knowledge
storage and retrieval for *documents*; this project extends it to *experience
nobody wrote down*.

## Goals

- One memory store shared by all MCP-speaking agents on the machine.
- Hybrid write model: memories are auto-extracted from transcripts
  (landing as `proposed`) and promoted by usage or explicit confirmation.
- Three scopes: `user` (follows the user everywhere), `project` (bound to a
  repo), `shared` (explicit workspace id for concurrent agents).
- Fully local: Ollama models, on-disk storage, nothing leaves the machine.
- Production-grade quality: clean module boundaries, tests first-class,
  versioned migrations, structured logging, honest error handling.

## Non-goals (v1)

- Team/multi-user features, sync, auth, ACLs, admin UI.
- Cloud model providers (interface exists; only Ollama implemented).
- Detecting whether a recalled memory was truly *acted on* (v1 uses
  retrieval frequency as the usage signal; schema reserves `times_used`).

## What Recall already provides (reused)

- MCP server with SSE transport (`mcp_server.py`) — memory tools are added
  alongside `list_indexes` / `query_rag`.
- Hybrid retrieval: BM25 + Chroma vectors + AutoMerging, multi-index
  (`utils/hybrid_retriever.py`, `utils/multi_index_retriever.py`).
- Ollama wiring: `nomic-embed-text` embeddings, `gemma4` LLM.
- Chroma storage layer (`utils/chroma_client.py`).
- Change-detection pattern: file hashing + processed-log
  (`utils/ingest_tracker.py`) — reused for transcript idempotency.
- FastAPI app, browser UI, streaming; pytest suite and conventions.

## New components

### 1. Memory data model

SQLite is the source of truth; Chroma holds only the embedding for semantic
search (linked by memory id). BM25 runs over the same rows for hybrid recall.

Memory record:

| Field | Notes |
|---|---|
| `id` | uuid |
| `content` | one distilled fact, human-readable |
| `category` | `preference` \| `decision` \| `fact` \| `gotcha` |
| `scope_type` | `user` \| `project` \| `shared` |
| `scope_key` | `*` for user; git remote URL (fallback: repo root path) for project; explicit workspace id for shared |
| `status` | `proposed` → `confirmed` → `archived` |
| `confidence` | float from extractor or 1.0 for explicit `remember` |
| `source_harness` | e.g. `claude-code` |
| `source_session` | session/transcript id |
| `evidence` | verbatim transcript quote justifying the memory |
| `created_at`, `last_retrieved` | timestamps |
| `times_retrieved`, `times_used` | lifecycle fuel (`times_used` reserved) |
| `superseded_by` | id of newer contradicting memory, else null |

Additional table: `audit_log` — every read, write, supersedence, deletion
(what, when, by which tool/session). Powers debuggability and trust.

Rules:
- Nothing is hard-deleted except explicit `forget`; decay archives.
- Contradictions form chains via `superseded_by`; recall returns chain heads
  only, history remains inspectable.

### 2. MCP interface

Four tools added to the existing Recall MCP server:

1. `recall(query, scope?)` — hybrid search over memories; results include
   status and provenance; `proposed` memories included but labeled.
2. `remember(content, category?, scope?)` — explicit write, lands `confirmed`.
3. `forget(memory_id)` — hard delete, audit-logged.
4. `memory_briefing(project_path)` — session-start digest: top confirmed
   facts, recent decisions, active gotchas, and any flagged conflicts for
   the resolved project + user scope. Triggered by a Claude Code
   SessionStart hook or by tool-description nudge in other harnesses.

Design notes:
- Retrieval implicitly increments `times_retrieved` (feeds promote/decay).
- Tool descriptions are load-bearing (they determine when agents call the
  tools) and are expected to be iterated.
- Existing document tools are untouched; documents and memories coexist.

### 3. Extractor (transcript → proposed memories)

Trigger: Claude Code `SessionEnd` hook pings Recall's API with the transcript
path. Fallback: a watcher polls known transcript directories
(`~/.claude/projects/**/*.jsonl`) using the hash-tracking pattern.
Extraction is asynchronous and never blocks agents.

Pipeline stages:
1. **Parse** — JSONL → clean user/assistant turns; tool calls summarized.
2. **Redact** — scrub credentials/tokens/keys (regex patterns + entropy
   check) *before* any LLM call. Non-negotiable: transcripts contain
   secrets; a memory store must never memorize them.
3. **Distill** — local LLM (gemma4) with a structured-output prompt:
   extract durable facts worth remembering across sessions. Each candidate:
   `{content, category, scope, confidence, evidence_quote}`.
4. **Gate** — quality filters; most candidates should die here:
   - **Grounding:** `evidence_quote` must appear verbatim in the transcript,
     else the candidate is a hallucination — rejected.
   - **Durability:** reject session-local facts, facts derivable from the
     code itself, and vague observations. Keep decisions-with-reasons,
     preferences, environment quirks, gotchas.
   - **Budget:** max ~5 memories per session.
   - **Confidence floor:** below threshold → dropped entirely.
5. **Write** — survivors land as `proposed` with full provenance;
   near-duplicates are routed to the lifecycle engine instead of written.

Idempotency: each transcript hash is processed once. Failures are logged and
retryable; a failed extraction loses at most one session's candidate
memories and never corrupts existing data.

Expectation: pipeline code is quick; the prompt + gates are the long-tail
iteration target, driven by the extraction eval set (below).

### 4. Lifecycle engine

Memories compete for existence.

- **Dedup (write time):** embed candidate, compare within scope.
  Very similar → *reinforce* existing memory (bump confidence, attach new
  provenance) instead of writing. Gray zone → small LLM-judge call
  classifying *same / contradicts / distinct*. Distinct → write.
- **Contradictions:** new memory written; old gets `superseded_by`.
  Safety rule: a `proposed` memory never silently supersedes a `confirmed`
  one — the pair is flagged *conflicted* and surfaced in the briefing for
  user resolution.
- **Promote/decay (periodic sweep):**
  - Promote `proposed` → `confirmed` after retrieval in N distinct sessions
    (start N=3) or explicit confirmation.
  - Decay: relevance score ~ recency of last retrieval × usage; below floor
    → `archived` (hidden from recall, visible in inspector, restorable).

### 5. Inspector

CLI first (`recall memory list | show <id> | confirm | forget | conflicts`),
showing full provenance chains. Browser-UI memory tab later. Rationale:
memory without visibility is creepy; with it, trustworthy and debuggable.

## Quality bar

- Clean interfaces: `MemoryStore` (SQLite impl) and `ModelProvider`
  (Ollama impl) so internals are swappable and testable in isolation.
- Versioned schema migrations from the first table.
- All config centralized (extends `config.py`); no scattered constants.
- Structured logging; no silent exception swallowing.
- Secrets redaction in the extractor from day one.

## Testing strategy

- Unit tests per component: store, gates, dedup thresholds, decay math,
  scope resolution (pure logic, easily testable).
- **Extraction eval set:** sample transcripts + expected memories, scored in
  pytest — the harness used to tune the distill prompt and gates.
- End-to-end: fake transcript → hook fires → memory appears as `proposed` →
  `recall` finds it → `memory_briefing` includes it.

## Build order

1. Data model + migrations + `MemoryStore` (SQLite) + audit log
2. MCP tools (`remember`/`recall` manual path working end-to-end)
3. `memory_briefing` + SessionStart hook integration
4. Extractor pipeline + SessionEnd hook + eval set
5. Lifecycle engine (dedup → contradictions → promote/decay)
6. Inspector CLI

Each step ships something usable; the manual path (step 2) already delivers
cross-agent memory before any auto-extraction exists.

## Risks

- **Extraction quality** is the make-or-break; mitigated by the grounding
  gate, budget, and the eval set from day one.
- **Small-model ceiling:** gemma4 may under-perform on distillation;
  `ModelProvider` interface allows swapping models per stage.
- **Harness coverage:** hooks are Claude Code-specific; the transcript
  watcher is the generic fallback, but other harnesses' transcript formats
  will need per-format parsers over time.
