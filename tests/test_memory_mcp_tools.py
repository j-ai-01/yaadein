import json
import math
from yaadein.mcp_tools import memory_tool_definitions, handle_memory_tool, is_memory_tool
from yaadein.service import MemoryService
from yaadein.store import MemoryStore
from yaadein.vector_index import MemoryVectorIndex


class FakeEmbedder:
    _axes = ["pytest", "deploy", "auth", "coffee", "kyun"]

    def embed(self, text):
        words = text.lower()
        vec = [1.0 if axis in words else 0.01 for axis in self._axes]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def make_service(tmp_path):
    return MemoryService(
        store=MemoryStore(tmp_path / "memories.db"),
        vector_index=MemoryVectorIndex(
            chroma_dir=tmp_path / "chroma",
            embedder=FakeEmbedder(),
            collection_name="test_memories",
        ),
        episode_index=MemoryVectorIndex(
            chroma_dir=tmp_path / "chroma_ep",
            embedder=FakeEmbedder(),
            collection_name="test_episodes",
        ),
    )


def test_tool_definitions_expose_six_tools():
    names = {t.name for t in memory_tool_definitions()}
    assert names == {"remember", "recall_memory", "forget_memory",
                     "memory_briefing", "recall_conversations", "read_conversation"}


def test_non_memory_tool_returns_none(tmp_path):
    assert handle_memory_tool("query_rag", {}, make_service(tmp_path)) is None


def test_remember_then_recall_roundtrip(tmp_path):
    service = make_service(tmp_path)
    remembered = json.loads(handle_memory_tool(
        "remember",
        {"content": "User prefers pytest", "category": "preference"},
        service,
    ))
    assert remembered["status"] == "confirmed"

    recalled = json.loads(handle_memory_tool(
        "recall_memory", {"query": "pytest testing"}, service
    ))
    assert recalled[0]["content"] == "User prefers pytest"


def test_forget_memory_reports_result(tmp_path):
    service = make_service(tmp_path)
    remembered = json.loads(handle_memory_tool(
        "remember", {"content": "temp fact"}, service
    ))
    result = json.loads(handle_memory_tool(
        "forget_memory", {"memory_id": remembered["id"]}, service
    ))
    assert result == {"forgotten": True}


def test_memory_briefing_returns_sections(tmp_path):
    service = make_service(tmp_path)
    handle_memory_tool(
        "remember", {"content": "User prefers pytest", "category": "preference"},
        service,
    )
    briefing = json.loads(handle_memory_tool("memory_briefing", {}, service))
    assert set(briefing) == {
        "facts", "decisions", "gotchas", "conflicts", "recent_conversations",
    }
    assert briefing["facts"][0]["content"] == "User prefers pytest"


def test_missing_required_argument_returns_error(tmp_path):
    result = json.loads(handle_memory_tool("remember", {}, make_service(tmp_path)))
    assert "error" in result


def test_is_memory_tool_true_for_memory_tools():
    for name in ("remember", "recall_memory", "forget_memory", "memory_briefing"):
        assert is_memory_tool(name) is True


def test_is_memory_tool_false_for_query_rag():
    assert is_memory_tool("query_rag") is False


def test_conversation_roundtrip_via_tools(tmp_path):
    service = make_service(tmp_path)
    ep = service.record_episode(
        summary="Designed Kyun provenance.", excerpt="USER: kyun idea...",
        scope_type="user", scope_key="*",
    )
    hits = json.loads(handle_memory_tool(
        "recall_conversations", {"query": "what did we say about kyun?"}, service))
    assert hits[0]["id"] == ep.id
    detail = json.loads(handle_memory_tool(
        "read_conversation", {"episode_id": ep.id}, service))
    assert detail["excerpt"].startswith("USER: kyun")
    assert detail["fact_ids"] == []


def test_read_conversation_unknown_id_returns_error(tmp_path):
    result = json.loads(handle_memory_tool(
        "read_conversation", {"episode_id": "ep_nope"}, make_service(tmp_path)))
    assert "unknown episode" in result["error"]


def test_unexpected_service_exception_returns_json_error(tmp_path):
    """A non-KeyError exception raised by the service is caught and returned as a JSON error, not raised."""
    class ExplodingService:
        def recall(self, query, project_key=None):
            raise RuntimeError("index unavailable")

    result = json.loads(handle_memory_tool(
        "recall_memory", {"query": "pytest"}, ExplodingService()))
    assert result == {"error": "index unavailable"}


def test_recall_conversations_missing_query_returns_error(tmp_path):
    result = json.loads(handle_memory_tool(
        "recall_conversations", {}, make_service(tmp_path)))
    assert "error" in result
