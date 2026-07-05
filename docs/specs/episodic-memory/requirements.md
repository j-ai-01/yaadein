# Requirements: Episodic Memory (Yaadein v2)

**Date:** 2026-07-05 · **Status:** for approval · **Upstream:** research.md
(findings F1–F11, decisions Q1–Q3) · **Downstream:** design.md

## Purpose

Yaadein SHALL remember conversations, not only facts: agents can ask "what
did we discuss about X?", get ranked conversation summaries, and drill into
the verbatim (redacted) excerpt behind any of them.

## Definitions

- **Episode** — a write-once record of one extraction pass's conversation
  window: an LLM summary, a redacted verbatim excerpt, a transcript pointer
  (path + turn range), scope, session/harness provenance, and the ids of
  facts extracted in the same pass.
- **Excerpt** — the rendered "ROLE: text" turns of the window, redacted,
  stored at episode creation (Q2: hybrid — stored copy + live pointer).

## Requirements

### R1 — Episode capture *(← F9, F10, Q2)*

- R1.1 WHEN an extraction pass completes over a non-empty turn window, THE
  SYSTEM SHALL persist exactly one episode containing: summary, redacted
  excerpt, transcript path, turn range, session id, harness, scope, and
  creation time.
- R1.2 WHEN the turn window is empty (no new turns), THE SYSTEM SHALL NOT
  create an episode.
- R1.3 THE excerpt SHALL be redacted with the same scrubber as the distill
  input BEFORE persistence; raw unredacted text SHALL NOT be stored.

### R2 — Summary quality *(← F11, and the nameless-Kyun incident)*

- R2.1 THE summary SHALL be 3–5 sentences derived only from the pass's raw
  turns (never from prior summaries — F3).
- R2.2 THE summary prompt SHALL require preservation of proper names
  (projects, tools, people) appearing in the window.

### R3 — Fact ↔ episode linkage *(← F1, F4)*

- R3.1 WHEN facts are written or reinforced during a pass that produced an
  episode, THE SYSTEM SHALL record that episode's id on each new fact.
- R3.2 WHEN an episode is retrieved, THE SYSTEM SHALL be able to list the
  ids of facts created in its pass.
- R3.3 Facts created via explicit `remember` (no episode) SHALL carry a null
  episode reference.

### R4 — Conversation search *(← F7, Q1)*

- R4.1 THE SYSTEM SHALL expose an MCP tool `recall_conversations(query,
  project_path?)` returning ranked episodes, each with: episode id, summary,
  created_at, session id, and scope.
- R4.2 Ranking SHALL combine semantic similarity of the summary with a
  recency weight (more recent episodes rank higher at equal similarity).
- R4.3 Scope filtering SHALL match fact recall: user-scope episodes always;
  project-scope episodes only for the matching project key.
- R4.4 Search SHALL operate over summaries only in v2 (raw-span search is
  out of scope — Q1).

### R5 — Drill-down *(← F3, F5, Q1, Q2)*

- R5.1 THE SYSTEM SHALL expose an MCP tool `read_conversation(episode_id)`
  returning the episode's summary, redacted excerpt, transcript pointer,
  session id, created_at, and linked fact ids.
- R5.2 WHEN the source transcript has been deleted, THE stored excerpt SHALL
  still be returned (episodes never go blind).
- R5.3 WHEN the episode id is unknown, THE tool SHALL return a JSON error,
  not raise.

### R6 — Briefing integration *(← Q3)*

- R6.1 `memory_briefing` SHALL include a `recent_conversations` section with
  at most 3 most-recent in-scope episodes (id, first sentence of summary,
  created_at).
- R6.2 THE briefing SHALL remain free of LLM and embedding calls.

### R7 — Write-once episodes *(← F3)*

- R7.1 Episodes SHALL never be updated or re-summarized after creation.
  Consolidation across episodes is out of scope (Sapne).

### R8 — Migration & compatibility

- R8.1 Schema change SHALL ship as migration v2; an existing v1 database
  SHALL migrate losslessly (all memories, audit rows preserved).
- R8.2 All v1 tools and behaviors SHALL be unchanged; existing tests SHALL
  pass unmodified except where they assert the schema version.

### R9 — Failure semantics

- R9.1 WHEN summary generation fails (LLM error/unparseable), THE SYSTEM
  SHALL NOT mark the transcript processed and SHALL NOT write facts from
  that pass (the pass is atomic-retryable; order: distill facts → summarize
  → gate → write facts → write episode → mark processed).
- R9.2 WHEN episode persistence fails after facts were written, THE SYSTEM
  SHALL report the error and NOT mark processed; the retry MAY reinforce the
  already-written facts (accepted, bounded by dedup).
- R9.3 Every episode mutation SHALL be audit-logged.

### R10 — Privacy

- R10.1 All episode data SHALL remain local; excerpts SHALL pass redaction
  (R1.3); `read_conversation` SHALL never read files outside configured
  watch roots or the extract request's resolved transcript path.

## Non-functional

- N1: `recall_conversations` adds at most one embedding call per query.
- N2: Episode storage growth at heavy use SHALL stay under ~100 MB/year
  (excerpt ≤ `MEMORY_EPISODE_EXCERPT_MAX_CHARS`, default 6000 chars).
- N3: All new behavior covered by offline tests (no Ollama), per house rule.

## Out of scope (v2)

Raw-span semantic search · episode consolidation/reflection (Sapne) ·
lifecycle promote/decay for episodes (Plan 3) · Kiro transcript parser ·
briefing beyond R6.

## Traceability

| Req | Sources |
|---|---|
| R1 | F9, F10, Q2 |
| R2 | F11, F3 |
| R3 | F1, F4 |
| R4 | F7, Q1 |
| R5 | F3, F5, Q1, Q2 |
| R6 | Q3 |
| R7 | F3 |
| R8–R10, N1–N3 | house rules (local-first, TDD, audit-everything) |
