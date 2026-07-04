from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Memory:
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
        return asdict(self)


@dataclass
class Candidate:
    content: str
    category: str
    scope: str  # "user" | "project"
    confidence: float
    evidence_quote: str
