"""Content hashing used to detect whether a transcript has changed since it was last extracted."""

import hashlib
from pathlib import Path


def file_hash(path: Path) -> str:
    """Return the MD5 hex digest of the file at `path`, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
