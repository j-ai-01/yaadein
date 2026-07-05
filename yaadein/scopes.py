"""Resolves a filesystem path to a stable project identity for scoping
memories: prefer the git remote URL, fall back to the repo root, then to the
raw resolved path — so project-scoped memories follow the project even if
it's cloned to a different location.
"""

import subprocess
from pathlib import Path

USER_SCOPE_KEY = "*"


def _git_output(args, cwd: str) -> str:
    """Run a git command in `cwd`, returning stripped stdout or "" on any failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _normalize_remote(url: str) -> str:
    """Strip trailing slash and .git suffix so equivalent remote URLs compare equal."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def resolve_project_key(path: str) -> str:
    """Derive a stable scope_key for project-scoped memories: git remote URL if
    available, else the git repo root, else the resolved path itself."""
    resolved = str(Path(path).resolve())
    remote = _git_output(["remote", "get-url", "origin"], cwd=resolved)
    if remote:
        return _normalize_remote(remote)
    root = _git_output(["rev-parse", "--show-toplevel"], cwd=resolved)
    if root:
        return str(Path(root).resolve())
    return resolved
