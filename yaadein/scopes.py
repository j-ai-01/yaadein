import subprocess
from pathlib import Path

USER_SCOPE_KEY = "*"


def _git_output(args, cwd: str) -> str:
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
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def resolve_project_key(path: str) -> str:
    resolved = str(Path(path).resolve())
    remote = _git_output(["remote", "get-url", "origin"], cwd=resolved)
    if remote:
        return _normalize_remote(remote)
    root = _git_output(["rev-parse", "--show-toplevel"], cwd=resolved)
    if root:
        return str(Path(root).resolve())
    return resolved
