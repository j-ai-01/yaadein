"""Quality gates the extractor's raw LLM candidates must pass before becoming
memories: this is the hallucination defense for the pipeline. Enforces
grounding (the evidence quote must appear verbatim in the transcript),
per-session budget, a confidence floor, valid category/scope, content length
bounds, and in-batch de-duplication.
"""

from typing import List

from config import MEMORY_CONFIDENCE_FLOOR, MEMORY_MAX_PER_SESSION
from yaadein.types import Candidate

VALID_CATEGORIES = {"preference", "decision", "fact", "gotcha"}
VALID_SCOPES = {"user", "project"}
_MIN_CONTENT_CHARS = 10
_MAX_CONTENT_CHARS = 300


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so grounding/dedup comparisons ignore
    formatting differences."""
    return " ".join(text.split()).lower()


def _passes(candidate: Candidate, normalized_transcript: str) -> bool:
    """Whether one candidate clears every individual gate: valid category and
    scope, confidence at or above the floor, content within length bounds, and
    its evidence quote actually appearing in the transcript (grounding)."""
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
    """Filter and cap raw extraction candidates: drop any that fail `_passes`,
    drop in-batch duplicates by normalized content, keep only the top
    MEMORY_MAX_PER_SESSION survivors ranked by confidence."""
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
