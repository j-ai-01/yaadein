import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel

from config import MEMORY_WATCH_INTERVAL_SECONDS, MEMORY_WATCH_ROOT
from yaadein.mcp_tools import handle_memory_tool, is_memory_tool, memory_tool_definitions
from yaadein.service import get_memory_service
from yaadein.watcher import find_recent_transcripts, sniff_project_path

app = FastAPI(title="Yaadein")


@app.on_event("startup")
async def start_transcript_watcher():
    """Near-real-time extraction: periodically re-mine active transcripts.

    Safe because extraction is idempotent per content hash and near-duplicate
    facts reinforce instead of duplicating. Complements the SessionEnd hook,
    which still gives the immediate final pass when a session closes.
    """
    if MEMORY_WATCH_INTERVAL_SECONDS <= 0:
        return

    async def watch_loop():
        log = logging.getLogger(__name__)
        while True:
            await asyncio.sleep(MEMORY_WATCH_INTERVAL_SECONDS)
            try:
                candidates = find_recent_transcripts(
                    MEMORY_WATCH_ROOT, MEMORY_WATCH_INTERVAL_SECONDS * 2
                )
                for transcript in candidates:
                    await asyncio.to_thread(
                        _run_extraction,
                        str(transcript),
                        sniff_project_path(transcript),
                        transcript.stem,
                        "claude-code",
                    )
            except Exception:
                log.exception("transcript watcher cycle failed")

    asyncio.create_task(watch_loop())


@app.get("/health")
async def health():
    return {"status": "ok"}


class ExtractRequest(BaseModel):
    transcript_path: str
    project_path: Optional[str] = None
    session_id: Optional[str] = None
    harness: str = "claude-code"


def _run_extraction(
    transcript_path: str,
    project_path: Optional[str],
    session_id: Optional[str],
    harness: str,
) -> None:
    from yaadein.extractor import build_extractor

    log = logging.getLogger(__name__)
    try:
        result = build_extractor().extract(
            Path(transcript_path),
            source_harness=harness,
            project_path=project_path,
            session_id=session_id,
        )
        level = logging.WARNING if result.error else logging.INFO
        log.log(
            level,
            "extraction for %s: %d written, %d reinforced, error=%s",
            transcript_path, len(result.written), len(result.reinforced), result.error,
        )
    except Exception:
        log.exception("memory extraction failed for %s", transcript_path)


@app.post("/memory/extract")
async def memory_extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    path = Path(req.transcript_path).expanduser().resolve()
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Transcript not found: {req.transcript_path}"},
        )
    background_tasks.add_task(
        _run_extraction, str(path), req.project_path, req.session_id, req.harness
    )
    return {"status": "queued", "transcript": str(path)}


server = Server("yaadein")
sse = SseServerTransport("/messages")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return memory_tool_definitions()


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if is_memory_tool(name):
        try:
            service = get_memory_service()
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Memory service unavailable: {e}"}),
            )]
        result = handle_memory_tool(name, arguments or {}, service)
        return [types.TextContent(type="text", text=result)]
    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


@app.get("/sse")
async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


# Mounted as a raw ASGI app: handle_post_message sends its own HTTP response,
# so it must own the connection — a FastAPI route here double-sends and kills
# the MCP handshake (RuntimeError: response already completed).
app.mount("/messages", sse.handle_post_message)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host="127.0.0.1", port=8899)
