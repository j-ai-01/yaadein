import json
import os
import time

from yaadein.watcher import find_recent_transcripts, sniff_project_path


def make_transcript(root, project, name, age_seconds, now):
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / name
    path.write_text("{}")
    stamp = now - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_finds_only_recently_modified(tmp_path):
    now = time.time()
    fresh = make_transcript(tmp_path, "proj-a", "fresh.jsonl", age_seconds=60, now=now)
    make_transcript(tmp_path, "proj-a", "stale.jsonl", age_seconds=7200, now=now)
    result = find_recent_transcripts(tmp_path, active_within_seconds=600, now=now)
    assert result == [fresh]


def test_ignores_non_jsonl_and_missing_root(tmp_path):
    now = time.time()
    project_dir = tmp_path / "proj-a"
    project_dir.mkdir()
    (project_dir / "notes.txt").write_text("not a transcript")
    assert find_recent_transcripts(tmp_path, active_within_seconds=600, now=now) == []
    assert find_recent_transcripts(tmp_path / "nope", active_within_seconds=600) == []


def test_sniffs_cwd_from_transcript_entries(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"type": "attachment"}) + "\n"
        + json.dumps({"type": "user", "cwd": "/Users/jai/workplace/repo"})
    )
    assert sniff_project_path(path) == "/Users/jai/workplace/repo"


def test_sniff_returns_none_when_no_cwd(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {}}))
    assert sniff_project_path(path) is None
