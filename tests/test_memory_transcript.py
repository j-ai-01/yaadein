import json
from yaadein.transcript import (
    Turn,
    parse_codex_transcript,
    parse_transcript,
    transcript_text,
)


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


def test_parser_registry_knows_claude_jsonl():
    from yaadein.transcript import get_parser, parse_codex_transcript, parse_transcript
    assert get_parser("claude-jsonl") is parse_transcript
    assert get_parser("codex-jsonl") is parse_codex_transcript
    assert get_parser("kiro-sessions") is None


def test_parses_codex_response_items_once(tmp_path):
    p = tmp_path / "codex.jsonl"
    write_jsonl(p, [
        {"type": "session_meta", "payload": {"cwd": "/repo"}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "internal instructions"}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>skip</environment_context>"}],
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Can Yaadein read Codex sessions?"}],
        }},
        {"type": "event_msg", "payload": {
            "type": "user_message",
            "message": "Can Yaadein read Codex sessions?",
        }},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "memory_briefing",
            "arguments": "{}",
        }},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Yes, with a Codex parser."}],
        }},
        {"type": "event_msg", "payload": {
            "type": "agent_message",
            "message": "Yes, with a Codex parser.",
        }},
        {"type": "response_item", "payload": {
            "type": "reasoning",
            "encrypted_content": "secret-ish internal blob",
        }},
    ])
    assert parse_codex_transcript(p) == [
        Turn("user", "Can Yaadein read Codex sessions?"),
        Turn("assistant", "Yes, with a Codex parser."),
    ]


def test_non_dict_block_in_assistant_content_is_skipped(tmp_path):
    """A non-dict entry inside assistant content blocks is ignored rather than crashing."""
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        assistant_blocks(["not-a-dict", {"type": "text", "text": "still works"}]),
    ])
    assert parse_transcript(p) == [Turn("assistant", "still works")]


def test_message_content_neither_str_nor_list_is_skipped(tmp_path):
    """An entry whose message content is neither a string nor a list produces no turn."""
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        {"type": "user", "message": {"role": "user", "content": 42}},
        user_str("real question"),
    ])
    assert parse_transcript(p) == [Turn("user", "real question")]


def test_codex_content_neither_str_nor_list_returns_empty():
    """_text_from_codex_content returns "" for content that is neither str nor list."""
    from yaadein.transcript import _text_from_codex_content
    assert _text_from_codex_content(123, {"input_text"}) == ""


def test_codex_content_string_is_stripped():
    """_text_from_codex_content strips a bare string content payload."""
    from yaadein.transcript import _text_from_codex_content
    assert _text_from_codex_content("  hello  ", {"input_text"}) == "hello"


def test_codex_content_skips_non_dict_and_wrong_type_blocks():
    """_text_from_codex_content ignores non-dict blocks and blocks outside text_types."""
    from yaadein.transcript import _text_from_codex_content
    blocks = ["not-a-dict", {"type": "other_type", "text": "ignored"},
              {"type": "input_text", "text": "kept"}]
    assert _text_from_codex_content(blocks, {"input_text"}) == "kept"


def test_codex_parser_skips_malformed_json_lines(tmp_path):
    """parse_codex_transcript skips lines that fail JSON decoding."""
    p = tmp_path / "codex.jsonl"
    p.write_text(
        "not json at all\n"
        + json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }})
    )
    assert parse_codex_transcript(p) == [Turn("user", "hello")]


def test_extractor_rejects_unknown_format_gracefully(tmp_path):
    from yaadein.extractor import Extractor

    class NeverCalledGenerator:
        def generate(self, prompt):
            raise AssertionError("must not reach the LLM")

    transcript = tmp_path / "s.jsonl"
    transcript.write_text("{}")
    extractor = Extractor(
        service=None,  # never touched: format check happens first
        generator=NeverCalledGenerator(),
        extract_log=tmp_path / ".extracted.json",
    )
    result = extractor.extract(transcript, transcript_format="kiro-sessions")
    assert result.error is not None
    assert "kiro-sessions" in result.error
