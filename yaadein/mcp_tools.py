"""MCP tool surface for agents: definitions (name, description, JSON schema)
for the four memory tools plus the synchronous dispatch that turns a tool
call into a MemoryService call. Tool descriptions here are prompts the
connecting agent reads to decide when to call each tool, so their wording is
as load-bearing as the code.
"""

import json
import logging
from typing import Optional

from mcp import types

from yaadein.scopes import resolve_project_key
from yaadein.service import MemoryService

logger = logging.getLogger(__name__)

_MEMORY_TOOLS = {
    "remember", "recall_memory", "forget_memory", "memory_briefing",
    "recall_conversations", "read_conversation",
}


def is_memory_tool(name: str) -> bool:
    """Whether `name` is one of Yaadein's memory tools."""
    return name in _MEMORY_TOOLS


def memory_tool_definitions() -> list:
    """MCP tool definitions (name, description, input schema) for
    remember/recall_memory/forget_memory/memory_briefing, advertised to agents
    via list_tools. Descriptions are written for the calling agent, not humans —
    they steer when and how it invokes each tool."""
    return [
        types.Tool(
            name="recall_memory",
            description=(
                "Search the user's persistent cross-agent memory for preferences, "
                "past decisions, project conventions, and gotchas. Call this BEFORE "
                "assuming what the user prefers or how this project works. "
                "Pass project_path to include project-scoped memories."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path of the current project (optional).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="remember",
            description=(
                "Save a durable fact to the user's persistent memory, shared across "
                "all AI agents. Use for preferences, decisions with reasons, and "
                "project gotchas the user states or confirms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "One distilled fact."},
                    "category": {
                        "type": "string",
                        "enum": ["preference", "decision", "fact", "gotcha"],
                        "description": "Kind of fact (default: fact).",
                    },
                    "project_path": {
                        "type": "string",
                        "description": (
                            "If this fact is specific to a project, its absolute path; "
                            "omit for user-wide facts."
                        ),
                    },
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="forget_memory",
            description="Permanently delete a memory by id (from recall_memory results).",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory id to delete."},
                },
                "required": ["memory_id"],
            },
        ),
        types.Tool(
            name="memory_briefing",
            description=(
                "Get a session-start digest of what is known: top user preferences "
                "and facts, recent decisions, active gotchas, and unresolved "
                "conflicts. Call once at the start of a session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path of the current project (optional).",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="recall_conversations",
            description=(
                "Search past conversations by meaning. Use when the user refers "
                "to a prior discussion ('what did we discuss about…', 'that idea "
                "from last week'). Returns ranked episode summaries with dates; "
                "follow up with read_conversation for the full excerpt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."},
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path of the current project (optional).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="read_conversation",
            description=(
                "Fetch one past conversation's summary, verbatim excerpt, and "
                "linked memory ids, by episode id from recall_conversations or "
                "memory_briefing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "Episode id (ep_…) to read.",
                    },
                },
                "required": ["episode_id"],
            },
        ),
    ]


def handle_memory_tool(
    name: str, arguments: dict, service: MemoryService
) -> Optional[str]:
    """Synchronously execute a memory tool call and return its JSON-string
    result, or None if `name` isn't a memory tool. Errors (missing args or any
    exception from the service) are caught and returned as a JSON error object
    rather than raised, so a bad call never breaks the MCP session."""
    if name not in _MEMORY_TOOLS:
        return None
    try:
        return json.dumps(_dispatch(name, arguments, service))
    except KeyError as e:
        return json.dumps({"error": f"Missing required argument: {e.args[0]}"})
    except Exception as e:
        logger.exception("memory tool %s failed", name)
        return json.dumps({"error": str(e)})


def _project_key(arguments: dict) -> Optional[str]:
    """Resolve the tool call's optional project_path into a scope key, or None
    if the call didn't include one (i.e. user-scoped)."""
    path = arguments.get("project_path")
    return resolve_project_key(path) if path else None


def _dispatch(name: str, arguments: dict, service: MemoryService) -> object:
    """Route a tool name/arguments pair to the matching MemoryService call and
    return its (JSON-serializable) result."""
    if name == "remember":
        content = arguments["content"]
        project_key = _project_key(arguments)
        memory = service.remember(
            content=content,
            category=arguments.get("category", "fact"),
            scope_type="project" if project_key else "user",
            scope_key=project_key or "*",
        )
        return memory.to_dict()

    if name == "recall_memory":
        return service.recall(arguments["query"], project_key=_project_key(arguments))

    if name == "forget_memory":
        return {"forgotten": service.forget(arguments["memory_id"])}

    if name == "recall_conversations":
        return service.recall_episodes(
            arguments["query"], project_key=_project_key(arguments)
        )

    if name == "read_conversation":
        episode_id = arguments["episode_id"]
        detail = service.read_episode(episode_id)
        return detail if detail is not None else {"error": f"unknown episode: {episode_id}"}

    # memory_briefing
    return service.briefing(project_key=_project_key(arguments))
