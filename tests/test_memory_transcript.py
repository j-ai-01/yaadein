import json
from yaadein.transcript import Turn, parse_transcript, transcript_text


def write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries))


def user_str(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def assistant_blocks(blocks):
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}}


def test_parses_user_strings_and_assistant_text(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        {"type": "file-history-snapshot"},
        user_str("I prefer pytest over unittest"),
        assistant_blocks([
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Noted, pytest it is."},
        ]),
    ])
    turns = parse_transcript(p)
    assert turns == [
        Turn("user", "I prefer pytest over unittest"),
        Turn("assistant", "Noted, pytest it is."),
    ]


def test_tool_use_summarized_and_tool_results_skipped(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        assistant_blocks([
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "file1\nfile2"},
        ]}},
    ])
    turns = parse_transcript(p)
    assert turns == [Turn("assistant", "Checking. [tool: Bash]")]


def test_skips_malformed_lines_and_harness_injected_user_text(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "not json at all\n"
        + json.dumps(user_str("<system-reminder>ignore</system-reminder>")) + "\n"
        + json.dumps(user_str("real question"))
    )
    assert parse_transcript(p) == [Turn("user", "real question")]


def test_transcript_text_formats_roles():
    turns = [Turn("user", "hi"), Turn("assistant", "hello")]
    assert transcript_text(turns, max_chars=1000) == "USER: hi\nASSISTANT: hello"


def test_transcript_text_keeps_tail_when_over_budget():
    turns = [Turn("user", "old " * 50), Turn("assistant", "recent answer")]
    out = transcript_text(turns, max_chars=40)
    assert "recent answer" in out
    assert "old" not in out


def test_skips_entry_with_non_dict_message(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        {"type": "user", "message": "weird"},
        user_str("real question"),
    ])
    assert parse_transcript(p) == [Turn("user", "real question")]
