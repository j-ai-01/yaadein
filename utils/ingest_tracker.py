import json
from pathlib import Path


def load_ingested(log_path: Path) -> dict:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def save_ingested(ingested: dict, log_path: Path) -> None:
    log_path.write_text(json.dumps(ingested, indent=2))


def is_already_ingested(file_hash: str, ingested: dict) -> bool:
    return file_hash in ingested.values()
