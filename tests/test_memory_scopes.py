import subprocess
from yaadein.scopes import resolve_project_key, USER_SCOPE_KEY


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_user_scope_key_is_star():
    assert USER_SCOPE_KEY == "*"


def test_non_git_dir_resolves_to_absolute_path(tmp_path):
    assert resolve_project_key(str(tmp_path)) == str(tmp_path.resolve())


def test_git_repo_without_remote_resolves_to_repo_root(tmp_path):
    _git("init", cwd=tmp_path)
    subdir = tmp_path / "src"
    subdir.mkdir()
    assert resolve_project_key(str(subdir)) == str(tmp_path.resolve())


def test_git_repo_with_remote_resolves_to_normalized_url(tmp_path):
    _git("init", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/jai/recall.git", cwd=tmp_path)
    assert resolve_project_key(str(tmp_path)) == "https://github.com/jai/recall"


def test_missing_git_binary_falls_back_to_absolute_path(tmp_path, monkeypatch):
    import subprocess

    real_run = subprocess.run

    def fake_run(args, *a, **kw):
        if args and args[0] == "git":
            raise FileNotFoundError("git not found")
        return real_run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_project_key(str(tmp_path)) == str(tmp_path.resolve())
