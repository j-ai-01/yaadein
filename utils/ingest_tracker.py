"""JSON-file persistence for the extraction log (per-transcript hash + turn bookmark)."""

import json
from pathlib import Path


def load_ingested(log_path: Path) -> dict:
    """Load the extraction log from `log_path`, or {} if it doesn't exist yet."""
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def save_ingested(ingested: dict, log_path: Path) -> None:
    """Write the extraction log to `log_path` as pretty-printed JSON."""
    log_path.write_text(json.dumps(ingested, indent=2))


def is_already_ingested(file_hash: str, ingested: dict) -> bool:
    """Whether `file_hash` appears anywhere among the log's recorded hashes."""
    return file_hash in ingested.values()
