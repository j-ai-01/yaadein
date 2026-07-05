"""Transcript parsing for the extraction pipeline: turns a harness-specific
session log into a flat list of (role, text) Turns the LLM can read, plus the
PARSERS registry mapping a config-declared format name to its parser
function, and the tail-truncation helper that keeps transcripts within the
LLM's context window.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Turn:
    """One user or assistant utterance extracted from a transcript."""

    role: str  # "user" | "assistant"
    text: str


def _text_from_blocks(blocks, include_tools: bool) -> str:
    """Flatten a message's content blocks into one string, keeping only text
    blocks (and, for assistant turns, a `[tool: name]` marker for tool_use blocks)."""
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
        elif block.get("type") == "tool_use" and include_tools:
            parts.append(f"[tool: {block.get('name', 'unknown')}]")
    return " ".join(parts)


def parse_transcript(path: Path) -> List[Turn]:
    """Parse a Claude Code JSONL transcript into user/assistant Turns, skipping
    malformed lines, non-message entries, and harness-injected reminders
    (user text starting with "<")."""
    turns: List[Turn] = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict) or entry.get("type") not in ("user", "assistant"):
            continue
        message = entry.get("message") or {}
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        role = entry["type"]
        if isinstance(content, str):
            text = content.strip()
            if role == "user" and text.startswith("<"):
                continue  # harness-injected reminders/commands, not user speech
        elif isinstance(content, list):
            text = _text_from_blocks(content, include_tools=(role == "assistant"))
        else:
            continue
        if text:
            turns.append(Turn(role, text))
    return turns


def transcript_text(turns: List[Turn], max_chars: int) -> str:
    """Render turns as "ROLE: text" lines joined by newlines, tail-truncated to
    max_chars (keeping the most recent conversation, snapped to a line boundary)
    so the distillation prompt fits the LLM's context window."""
    lines = [f"{turn.role.upper()}: {turn.text}" for turn in turns]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    return tail[newline + 1:] if newline != -1 else tail


# Registry of transcript formats the extractor understands. Supporting a new
# harness = write a parser returning List[Turn], register it here, and add a
# watch source for it in config.py. Sources whose format has no parser yet
# are skipped with a warning instead of crashing.
PARSERS = {
    "claude-jsonl": parse_transcript,
    # "kiro-sessions": parse_kiro_sessions,  # pending real Kiro session data
}


def get_parser(format_name: str):
    """Look up the parser function registered for `format_name`, or None if
    no parser exists yet for that transcript format."""
    return PARSERS.get(format_name)
