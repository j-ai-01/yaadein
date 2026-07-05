# Design: Episodic Memory (Yaadein v2)

**Date:** 2026-07-05 · **Status:** for approval · **Upstream:**
requirements.md (R1–R10, N1–N3) · **Downstream:** tasks.md

## Overview

One new first-class record — the **episode** — stored in SQLite (truth),
indexed in a dedicated Chroma collection (meaning), linked bidirectionally
to facts, produced by the existing extraction pass, exposed through two new
MCP tools and a briefing section. No changes to v1 behavior.

## D1 — Schema: migration v2 *(R1, R3, R8)*

Append to `MIGRATIONS` in `yaadein/schema.py` (first real use of the
versioned-migration machinery):

```sql
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    source_harness TEXT,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('user','project','shared')),
    scope_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    excerpt TEXT NOT NULL,            -- redacted, capped (R1.3, N2)
    transcript_path TEXT,             -- provenance pointer (Q2 hybrid)
    transcript_format TEXT,
    turn_start INTEGER,
    turn_end INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_episodes_scope ON episodes (scope_type, scope_key);
CREATE INDEX idx_episodes_created ON episodes (created_at);

ALTER TABLE memories ADD COLUMN episode_id TEXT;
CREATE INDEX idx_memories_episode ON memories (episode_id);
```

- **Fact→episode link** is a nullable `episode_id` column (R3.3: null for
  explicit `remember`). **Episode→facts** (R3.2) is the reverse query
  `SELECT id FROM memories WHERE episode_id = ?` — *rejected alternative:*
  a junction table; overkill for a strict 1-N relationship.
- v1→v2 migration is additive only; existing rows untouched (R8.1). The
  only test allowed to change is the schema-version assertion (R8.2).

## D2 — Types & store *(R1, R9.3)*

- `Episode` dataclass in `types.py` mirroring the columns, plus `to_dict()`.
- `Memory` gains `episode_id: Optional[str] = None` (appended to `_COLUMNS`;
  row mapping is by name, so column order is not load-bearing).
- `MemoryStore` gains:
  - `add_episode(episode: Episode) -> Episode` — honors a **caller-preset
    id** (see D5); assigns uuid/created_at only if empty; audits
    `"add_episode"`.
  - `get_episode(episode_id) -> Optional[Episode]`
  - `list_episodes(scope_type=None, scope_key=None, limit=None)` — newest
    first (serves R6 without Chroma).
  - `fact_ids_for_episode(episode_id) -> List[str]`

## D3 — Meaning layer: second Chroma collection *(R4, F8)*

A second `MemoryVectorIndex` instance over collection
`yaadein_episodes` (config `MEMORY_EPISODE_COLLECTION`), embedding the
**summary only**. *Rejected alternative (F8 open choice):* one collection
with a `type` metadata filter — separate collections keep v1's collection
untouched (R8.2) and make wipes/rebuilds independent.

`MemoryService` gains an optional `episode_index=None` constructor kwarg —
v1 call sites and tests construct unchanged; episode features no-op safely
when absent (extractor skips episode creation, `recall_episodes` returns
`[]`). `get_memory_service()` wires both indexes.

## D4 — Retrieval & ranking *(R4.2, F7)*

`recall_episodes(query, project_key=None, top_k=5)`:

1. Over-fetch top-20 summary embeddings from Chroma.
2. Scope-filter via the same `_in_scope` rule as facts (R4.3).
3. Score = `similarity + MEMORY_EPISODE_RECENCY_WEIGHT × 0.5^(age_days / MEMORY_EPISODE_RECENCY_HALFLIFE_DAYS)`
   — additive exponential-decay bonus (Generative-Agents style, same shape
   as the existing keyword bonus). Defaults: weight `0.15`, half-life `7`
   days, both env-tunable. A week-old episode gets +0.075; a fresh one
   +0.15 — enough to break ties, never enough to drown similarity.
4. Sort, cut to top_k. No retrieval counters on episodes in v2 (lifecycle
   is Plan 3).

`read_episode(episode_id)` returns `episode.to_dict()` +
`fact_ids` (D2 reverse query). Serves R5 entirely from the DB — stored
excerpt means no transcript read, so deletion-blindness (R5.2) is
structurally impossible.

## D5 — Extractor changes *(R1, R2, R9)*

New second prompt `_SUMMARY_PROMPT` (3–5 sentences, names mandatory —
wording seeded from the successful spike, F11). *Rejected alternative:* one
combined facts+summary JSON response — small local models fail structured
output often enough (we ship an unparseable-output path already) that
coupling both artifacts to one parse doubles the blast radius.

Pass order (R9.1):

```
hash check → bookmark slice → render → redact
→ distill facts (LLM #1)        — failure ⇒ error result, retryable
→ parse candidates              — unparseable ⇒ error result, retryable
→ summarize (LLM #2)            — failure ⇒ error result, retryable
→ gates
→ episode_id = pre-generated uuid ("ep_…")
→ write facts, each stamped episode_id     (rollback-per-fact as today)
→ write episode WITH that preset id (store + chroma, rollback pair)
→ mark processed (hash + bookmark)
```

**Accepted corner (documented per R9.2):** if fact writes succeed but the
episode write fails, the pass is not marked processed; the retry generates
a *new* episode id, and the original facts (now reinforced, not rewritten)
keep a dangling `episode_id`. `read_episode` on a dangling id returns the
JSON error of R5.3; nothing crashes. Bounded, rare (episode write is two
local operations), and preferable to the alternatives: writing the episode
first would duplicate episodes on fact-write failure — worse, because
episodes are user-visible search results while a dangling id is invisible
until traced.

Episode skip conditions: empty window (R1.2) and `episode_index is None`
(back-compat, D3) — in the latter case facts get `episode_id=None`.

`ExtractionResult` gains `episode_id: Optional[str] = None`.

## D6 — MCP surface *(R4, R5, R6)*

- `recall_conversations` → `service.recall_episodes` (tool description:
  "search past conversations by meaning; use when the user refers to a
  prior discussion").
- `read_conversation` → `service.read_episode`; unknown id ⇒
  `{"error": "unknown episode: …"}` (R5.3).
- `_MEMORY_TOOLS` grows by both names (gates the lazy service init in
  `server.py` unchanged).
- `briefing()` gains `recent_conversations`: top 3 from
  `store.list_episodes(...)` — id, `summary.split(". ")[0]`, created_at.
  Pure SQL (R6.2).

## D7 — Config additions

```python
MEMORY_EPISODE_COLLECTION = "yaadein_episodes"
MEMORY_EPISODE_EXCERPT_MAX_CHARS = _env_int("EPISODE_EXCERPT_MAX_CHARS", 6000)   # N2
MEMORY_EPISODE_RECENCY_WEIGHT = _env_float("EPISODE_RECENCY_WEIGHT", 0.15)       # D4
MEMORY_EPISODE_RECENCY_HALFLIFE_DAYS = _env_float("EPISODE_RECENCY_HALFLIFE_DAYS", 7.0)
MEMORY_BRIEFING_LIMITS = {..., "conversations": 3}                               # R6.1
```

## D8 — Error handling summary

| Failure | Behavior | Req |
|---|---|---|
| Summary LLM call fails | error result, nothing written, retryable | R9.1 |
| Episode store/chroma write fails | error result, not marked; dangling-id corner (D5) | R9.2 |
| `read_conversation` unknown id | JSON error | R5.3 |
| Transcript deleted post-episode | stored excerpt still served | R5.2 |
| Service built without episode index | episode features no-op; facts unaffected | R8.2 |

## D9 — Testing strategy *(N3)*

- **Migration:** build a v1 DB (run only `MIGRATIONS[0]`), insert memories +
  audit rows, migrate → version 2, all rows intact, `episode_id` null (R8.1).
- **Store:** episode CRUD, preset-id honor, newest-first listing, reverse
  fact query, audit rows.
- **Service:** record/recall/read with FakeEmbedder; recency bonus ordering
  (two equal-similarity episodes, different ages); scope filtering; no-index
  no-op path.
- **Extractor:** SequencedGenerator (two LLM responses per pass: facts JSON,
  then summary text); episode created with facts stamped; R9.1 order —
  summary failure ⇒ no facts written, retryable; excerpt is redacted and
  capped.
- **Tools:** roundtrip recall_conversations → read_conversation; unknown id
  error; briefing shows recent_conversations.
- **E2E:** fake transcript → extract → episode searchable by meaning →
  drill-down returns excerpt containing a known phrase.
- **Eval (deferred to `-m eval`):** episode-summary quality cases seeded
  from the spike.

## Traceability

| Design | Satisfies |
|---|---|
| D1 | R1, R3, R8 |
| D2 | R1, R3, R9.3 |
| D3 | R4, R8.2, F8 |
| D4 | R4, R5, F7 |
| D5 | R1, R2, R9, F3, F11 |
| D6 | R4, R5, R6 |
| D7 | N2, R6.1, D4 |
| D8 | R5, R8, R9 |
| D9 | N3, R8.1 |
