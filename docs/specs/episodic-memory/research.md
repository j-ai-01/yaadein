# Research: Episodic Memory for Yaadein (v2)

**Date:** 2026-07-05 · **Status:** for review · **Next stage:** requirements.md

## Problem statement

Yaadein v1 stores atomized durable facts (semantic memory) and discards the
conversation that produced them. Discovered in production (2026-07-05): the
"Kyun" brainstorm was extracted as correct-but-nameless concept facts,
unfindable by the name binding them; "what did we discuss about X?" is
unanswerable. v2 must add conversation-level recall (episodic memory)
without losing v1's fast fact recall.

## Method — three parallel streams

1. **Prior-art survey** (web agent): Zep/Graphiti, MemGPT/Letta, mem0,
   Generative Agents (Park et al. 2023), plus summarization-pitfall papers.
2. **Data audit** (script over real transcripts on this machine).
3. **Feasibility spike** (gemma4 summarizing the real Kyun conversation).

## Findings

### From prior art (F1–F8)

- **F1 — Layers over one substrate, linked.** Episodic and semantic memory
  are universally modeled as linked layers over one store (Zep: subgraphs of
  one graph; Letta: tiers), not separate databases. *Implication: episodes
  and facts should reference each other (foreign keys), not merely coexist.*
- **F2 — No industry convergence on granularity; per-message dominates.**
  Zep and Letta store per-message episodes; coarse summaries appear only as
  a separately-triggered reflection/consolidation step. A single summary per
  conversation is NOT the dominant pattern anywhere surveyed.
- **F3 — Summarization drift is the named failure mode.** Extraction-only
  and summary-only designs lose rare, high-importance details; a controlled
  ablation (arXiv 2601.00821) found verbatim chunks beat both extracted
  facts and summaries for long-conversation QA.
- **F4 — Backlinks are the strongest mitigation.** Generative Agents'
  reflections cite the observation ids they derive from; Zep keeps
  episode↔entity indices. *Implication: every synthesized memory should
  carry pointers to its source spans — and facts should link to the episode
  they came from.*
- **F5 — Raw text must be a first-class retrieval target,** not archival
  fallback (Letta's recall memory; the verbatim-chunks result). *Implication:
  the episode layer should make raw transcript spans retrievable, not only
  LLM summaries.*
- **F6 — Episodic→semantic consolidation is manual/heuristic everywhere.**
  No surveyed system does it automatically well; mem0's own community cannot
  map their marketing taxonomy to code. *Implication: our deliberate
  extraction trigger (pass/session) is industry-normal; don't chase
  automatic consolidation in v2.*
- **F7 — Unbounded episode logs degrade retrieval.** The episode layer needs
  its own retrieval design (embedding index + recency weighting) rather than
  a flat log scanned at query time.
- **F8 (unverified) — Separate vector collection vs metadata filtering** is
  an open implementation choice; no surveyed system documents the trade-off.

### From the data audit (F9–F10)

- **F9 — Real extraction windows are small.** 10 transcripts, 8.6 MB,
  4 projects; top sessions run 138–678 turns; average turn ≈ 433 chars →
  a 16k-char extraction window holds ≈ 37 turns. Per-pass episodes are
  naturally topic-sized, not day-sized.
- **F10 — Storage is a non-issue.** Worst-case heavy usage ≈ 120 episodes/
  day ≈ 22 MB/year of summaries. Granularity must be chosen for retrieval
  quality only; cost is irrelevant. Raw-span duplication into the store is
  also affordable if design wants it.

### From the spike (F11)

- **F11 — gemma4 produces usable, name-preserving episode summaries.**
  On the real Kyun excerpt (32 turns, ~12k chars) it captured the project
  name, the git-notes decision, privacy tiers, and the episodic-layer plan
  in 4 coherent sentences. The core quality risk is retired. (Single-sample
  spike; the eval harness should grow episode-summary cases in Tasks.)

## What this changes about the naive design sketch

The pre-research sketch was "one summary per extraction pass + transcript
pointer." Findings adjust it:

1. **Keep per-pass summaries** (F9 shows passes are topic-sized; F11 shows
   quality holds) — but treat them as *one* retrieval layer, not the only one.
2. **Add raw-span retrieval** (F3/F5): an episode must be able to return its
   verbatim transcript excerpt on demand — summaries for search, raw for
   truth. `read_conversation` is therefore a requirement, not a nicety.
3. **Link facts ↔ episodes** (F1/F4): facts extracted in a pass should carry
   the episode id of that pass, so any fact can answer "what conversation is
   this from?" and any episode can list its facts.
4. **Episode retrieval needs recency weighting** (F7), not just cosine rank.
5. **Never re-summarize summaries** (F3 drift): episodes are write-once from
   raw turns; consolidation-of-episodes is Sapne's problem, explicitly out
   of v2 scope.

## Open questions for the requirements review

- **Q1:** Should `recall_conversations` blend episode summaries and raw-span
  matches in one result list, or expose two tools? (F5 vs API simplicity.)
- **Q2:** Store raw excerpts in the DB (survives transcript deletion, F10
  says affordable; but duplicates sensitive text — must be redacted) or read
  them live from transcripts via pointers (fragile to deletion)? Hybrid:
  redacted excerpt stored, pointer kept for provenance?
- **Q3:** Do episodes participate in `memory_briefing` (recent-conversations
  section) in v2, or defer to keep briefing lean?

## Sources

Zep arXiv 2501.13956 · Graphiti docs · Letta agent-memory blog · MemGPT
paper · mem0 memory-types docs · mem0 GitHub issue #3644 · Generative
Agents (ACM 3586183.3606763) · arXiv 2601.00821 (verbatim chunks ablation) ·
arXiv 2603.07670 (agent memory survey) · local data audit + gemma spike
scripts (scratchpad, 2026-07-05).
