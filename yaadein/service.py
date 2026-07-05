"""The application layer that ties the SQLite store and the Chroma vector
index together: MemoryService is the single object both the MCP tools and
the extraction pipeline call into for every memory operation.

It owns the two cross-cutting rules that don't belong to either storage
layer: scope filtering (which memories a given project/user can see) and
hybrid ranking (semantic similarity from Chroma plus a keyword-overlap bonus,
since embeddings alone under-rank exact-term matches).
"""

from typing import List, Optional, Tuple

from config import (
    MEMORY_BRIEFING_LIMITS, MEMORY_CHROMA_DIR, MEMORY_COLLECTION,
    MEMORY_DB_PATH, MEMORY_KEYWORD_BONUS, MEMORY_TOP_K,
)
from yaadein.scopes import USER_SCOPE_KEY
from yaadein.store import MemoryStore
from yaadein.types import Memory
from yaadein.vector_index import MemoryVectorIndex

_KEYWORD_BONUS_CAP = 0.3


class MemoryService:
    """Coordinates the SQLite store (truth) and the Chroma index (semantic
    search) behind one API: remember/propose/recall/briefing/forget/
    find_similar/reinforce. Every write to the index is paired with a store
    write (or rolled back on index failure) so the two never drift apart."""

    def __init__(self, store: MemoryStore, vector_index: MemoryVectorIndex):
        self._store = store
        self._index = vector_index

    def remember(
        self,
        content: str,
        category: str = "fact",
        scope_type: str = "user",
        scope_key: str = USER_SCOPE_KEY,
        source_harness: Optional[str] = None,
        source_session: Optional[str] = None,
    ) -> Memory:
        """Save a fact directly as `confirmed` (used by the `remember` MCP tool,
        i.e. the user or agent explicitly asserted it — no gating needed).
        Rolls back the SQLite write if indexing fails, so the two stores stay in sync."""
        memory = Memory(
            id="", content=content, category=category,
            scope_type=scope_type, scope_key=scope_key,
            status="confirmed", confidence=1.0,
            source_harness=source_harness, source_session=source_session,
        )
        saved = self._store.add(memory)
        try:
            self._index.add(saved.id, saved.content)
        except Exception:
            self._store.forget(saved.id)
            raise
        return saved

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
        """Save a fact from the extraction pipeline as `proposed` (unconfirmed),
        carrying its confidence and evidence quote for later review. Same
        index/store rollback contract as `remember`."""
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
        """Look for an existing, non-archived memory in the same scope whose
        content is semantically close to `content`. Used by the extractor to
        decide reinforce-vs-propose for a new candidate; returns the closest
        in-scope match and its similarity, or None if none qualifies."""
        # over-fetch (mirrors recall's rationale) so that other-scope near-hits
        # don't crowd out an in-scope duplicate sitting further down the ranking
        for memory_id, similarity in self._index.query(content, top_k=20):
            memory = self._store.get(memory_id)
            if memory is None or memory.status == "archived" or memory.superseded_by:
                continue
            if memory.scope_type == scope_type and memory.scope_key == scope_key:
                return memory, similarity
        return None

    def reinforce(self, memory_id: str, source_session: Optional[str] = None) -> None:
        """Bump an existing memory's confidence instead of writing a duplicate;
        thin pass-through to the store, kept here so callers only ever talk to MemoryService."""
        self._store.reinforce(memory_id, source_session)

    def recall(
        self,
        query: str,
        project_key: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Hybrid-ranked search for the `recall_memory` tool: semantic similarity
        from Chroma plus a capped keyword-overlap bonus (so exact-term matches
        aren't buried by embeddings alone), filtered to memories visible at
        `project_key`'s scope and excluding archived/superseded ones. Records
        a retrieval for whatever is returned."""
        top_k = top_k or MEMORY_TOP_K
        # over-fetch so scope filtering still leaves top_k candidates
        hits = self._index.query(query, top_k=top_k * 4)
        terms = [t for t in query.lower().split() if len(t) > 3]

        scored = []
        for memory_id, similarity in hits:
            memory = self._store.get(memory_id)
            if memory is None:
                continue
            if memory.status == "archived" or memory.superseded_by:
                continue
            if not self._in_scope(memory, project_key):
                continue
            bonus = min(
                _KEYWORD_BONUS_CAP,
                MEMORY_KEYWORD_BONUS
                * sum(1 for t in terms if t in memory.content.lower()),
            )
            scored.append((memory, similarity + bonus))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:top_k]
        self._store.record_retrieval([m.id for m, _ in top])
        return [{**m.to_dict(), "score": round(score, 4)} for m, score in top]

    def forget(self, memory_id: str) -> bool:
        """Delete a memory from both the store and the vector index. Returns
        False if it didn't exist (index deletion is only attempted on success)."""
        removed = self._store.forget(memory_id)
        if removed:
            self._index.delete(memory_id)
        return removed

    def briefing(self, project_key: Optional[str] = None) -> dict:
        """Build the session-start digest for the `memory_briefing` tool: top
        confirmed facts/preferences by retrieval count, recent decisions,
        active gotchas (confirmed or proposed), and any unresolved conflicts —
        each capped per config.MEMORY_BRIEFING_LIMITS and scoped to `project_key`."""
        candidates = [
            m for m in self._store.list()
            if m.status != "archived"
            and not m.superseded_by
            and self._in_scope(m, project_key)
        ]

        def to_dict(memory: Memory) -> dict:
            d = memory.to_dict()
            if memory.status == "proposed":
                d["unconfirmed"] = True
            return d

        confirmed = [m for m in candidates if m.status == "confirmed"]
        facts = sorted(
            (m for m in confirmed if m.category in ("preference", "fact")),
            key=lambda m: m.times_retrieved, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["facts"]]
        decisions = sorted(
            (m for m in confirmed if m.category == "decision"),
            key=lambda m: m.created_at, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["decisions"]]
        gotchas = sorted(
            (m for m in candidates if m.category == "gotcha"),
            key=lambda m: m.created_at, reverse=True,
        )[: MEMORY_BRIEFING_LIMITS["gotchas"]]
        conflicts = [m for m in candidates if m.conflict_with]

        returned = facts + decisions + gotchas + conflicts
        self._store.record_retrieval(list(dict.fromkeys(m.id for m in returned)))
        return {
            "facts": [to_dict(m) for m in facts],
            "decisions": [to_dict(m) for m in decisions],
            "gotchas": [to_dict(m) for m in gotchas],
            "conflicts": [to_dict(m) for m in conflicts],
        }

    @staticmethod
    def _in_scope(memory: Memory, project_key: Optional[str]) -> bool:
        """Scope rule: user-scoped memories are always visible; project-scoped
        ones only when project_key matches; shared-scope is not yet reachable."""
        if memory.scope_type == "user":
            return True
        if memory.scope_type == "project":
            return project_key is not None and memory.scope_key == project_key
        return False  # shared scope arrives with the extractor/live workspaces


_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Process-wide singleton MemoryService, lazily constructing the store and
    vector index (and the Ollama embedder) on first use."""
    global _service
    if _service is None:
        from yaadein.vector_index import OllamaEmbedder

        _service = MemoryService(
            store=MemoryStore(MEMORY_DB_PATH),
            vector_index=MemoryVectorIndex(
                chroma_dir=MEMORY_CHROMA_DIR,
                embedder=OllamaEmbedder(),
                collection_name=MEMORY_COLLECTION,
            ),
        )
    return _service
