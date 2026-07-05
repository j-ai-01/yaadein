"""Core data shapes shared across the memory pipeline: the persisted `Memory`
record and the `Candidate` a not-yet-written extraction produces.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Memory:
    """A single memory record as stored in SQLite (see schema.py); the
    authoritative representation of a fact, its scope, status, and provenance."""

    id: str
    content: str
    category: str
    scope_type: str
    scope_key: str
    status: str = "proposed"
    confidence: float = 1.0
    source_harness: Optional[str] = None
    source_session: Optional[str] = None
    evidence: Optional[str] = None
    created_at: str = ""
    last_retrieved: Optional[str] = None
    times_retrieved: int = 0
    times_used: int = 0
    superseded_by: Optional[str] = None
    conflict_with: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON responses."""
        return asdict(self)


@dataclass
class Candidate:
    """An unconfirmed fact proposed by the extraction pipeline, before it passes
    the quality gates (gates.py) and is written to the store as `proposed`."""

    content: str
    category: str
    scope: str  # "user" | "project"
    confidence: float
    evidence_quote: str
