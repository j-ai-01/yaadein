import json
import time
from pathlib import Path
from typing import List, Optional


def find_recent_transcripts(
    root: Path, active_within_seconds: int, now: Optional[float] = None
) -> List[Path]:
    """Transcripts modified recently enough to be an active (or just-ended) session."""
    if not root.exists():
        return []
    if now is None:
        now = time.time()
    cutoff = now - active_within_seconds
    return sorted(
        path for path in root.glob("*/*.jsonl") if path.stat().st_mtime >= cutoff
    )


def sniff_project_path(transcript: Path, max_lines: int = 100) -> Optional[str]:
    """Pull the session's working directory from the transcript's own entries."""
    try:
        with open(transcript) as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(entry, dict):
                    cwd = entry.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
    except OSError:
        return None
    return None
