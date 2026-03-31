#!/usr/bin/env python3
"""
MCP Server: Agent-as-a-Service for the Agentic Research workflow.
Exposes the validated research flow (extract → retrieve → synthesis) as an MCP tool.

Run (default: HTTP/SSE on 0.0.0.0:8000 — MCP Inspector & Cursor SSE URL):
  uv run python src/mcp_server.py

Stdio (JSON-RPC for Cursor "command" MCP):
  uv run python src/mcp_server.py --stdio

FastMCP: ``transport="sse"`` serves ``/sse``; ``transport="http"`` serves ``/mcp`` (Streamable HTTP).

HITL (MMR λ): set ``AMI_HITL_DIVERSITY=1`` in ``.env``. When enough context chunks are retrieved,
the graph interrupts; call ``resume_research`` with the same ``thread_id`` and ``lambda_value``.

Dashboard SSE (Streamlit): ``GET /ami/dashboard/stream?query=...&thread_id=...`` and
``GET /ami/dashboard/resume/stream?thread_id=...&lambda_value=...`` — custom HTTP routes on the
same port as MCP SSE (e.g. :8000).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
os.chdir(_proj)

from dotenv import load_dotenv

_env_file = _proj / ".env"
# Prefer values from .env over inherited shell env (so AMI_HITL_* in .env actually applies).
load_dotenv(_env_file, override=True)

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from fastmcp import FastMCP
from langgraph.types import Command
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from app.graph.entity_workflow import build_workflow
from app.logic.workflow import make_initial_state
from app.services.telemetry import log_workflow_outcome
from app.ui.workflow_stream import iter_workflow_dashboard_events, sse_encode

mcp = FastMCP("Auto-Intelligence-Service")

_mcp_app = None


def get_mcp_workflow():
    """Single compiled graph + checkpointer so HITL interrupt/resume share state."""
    global _mcp_app
    if _mcp_app is None:
        if _hitl_enabled():
            from langgraph.checkpoint.memory import MemorySaver

            _mcp_app = build_workflow(checkpointer=MemorySaver())
        else:
            _mcp_app = build_workflow()
    return _mcp_app


def _hitl_enabled() -> bool:
    return os.environ.get("AMI_HITL_DIVERSITY", "").strip().lower() in ("1", "true", "yes")


def _log_mcp_startup_banner() -> None:
    """Print once so you can see whether HITL is actually on (must match .env after restart)."""
    if not _env_file.is_file():
        print(
            f"[MCP config] WARNING: no {_env_file} — copy .env.example to .env and set AMI_HITL_DIVERSITY=1",
            file=sys.stderr,
            flush=True,
        )
    hitl = _hitl_enabled()
    raw = os.environ.get("AMI_HITL_DIVERSITY", "")
    try:
        min_ctx = int(os.environ.get("AMI_HITL_MIN_CONTEXT", "3"))
    except ValueError:
        min_ctx = 3
    mode = (
        "HITL ON — research_company uses ainvoke until interrupt; resume_research streams (astream_events)"
        if hitl
        else "HITL OFF — research_company uses astream_events (no interrupt pause)"
    )
    print(
        f"[MCP config] AMI_HITL_DIVERSITY={'enabled' if hitl else 'disabled'} "
        f"(raw={raw!r}) | AMI_HITL_MIN_CONTEXT={min_ctx} | {mode}",
        file=sys.stderr,
        flush=True,
    )


def _mcp_progress(msg: str) -> None:
    """Progress lines to stderr — safe for stdio MCP (JSON-RPC uses stdout only)."""
    print(msg, file=sys.stderr, flush=True)


def _event_output_to_dict(output: Any) -> dict | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return None


def _normalize_state(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result) if result else {}


async def _merge_state_from_astream_events(
    app: Any,
    inputs: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the graph with astream_events (v2) and merge node outputs; log node starts to stderr.
    ``inputs`` may be initial state dict or ``Command(resume=...)``.
    """
    merged_state: dict[str, Any] = {}
    async for event in app.astream_events(inputs, config, version="v2"):
        kind = event.get("event")
        meta = event.get("metadata") or {}

        if kind == "on_chain_start" and meta.get("langgraph_node"):
            node_name = meta["langgraph_node"]
            _mcp_progress(f"  ──> Node start: {node_name}")
        elif kind == "on_tool_start":
            _mcp_progress(f"      Tool start: {event.get('name', '?')}")

        if kind == "on_chain_end":
            data = event.get("data") or {}
            out = _event_output_to_dict(data.get("output"))
            if out:
                merged_state.update(out)
    return merged_state


def _interrupt_response(rd: dict[str, Any], thread_id: str) -> str:
    intrs = rd.get("__interrupt__") or []
    serialized = []
    for it in intrs:
        if hasattr(it, "value"):
            serialized.append({"value": it.value, "id": getattr(it, "id", None)})
        else:
            serialized.append(str(it))
    return json.dumps(
        {
            "status": "interrupted",
            "thread_id": thread_id,
            "interrupt": serialized,
            "hint": "Call resume_research(thread_id=..., lambda_value=0.0-1.0) with the same thread_id.",
        },
        indent=2,
    )


@mcp.custom_route("/ami/dashboard/stream", methods=["GET"])
async def ami_dashboard_workflow_stream(request: Request) -> StreamingResponse | JSONResponse:
    """SSE of LangGraph node/tool events + final state (for Streamlit or other clients)."""
    query = (request.query_params.get("query") or "").strip()
    thread_id = (request.query_params.get("thread_id") or "dashboard").strip()
    if not query:
        return JSONResponse({"error": "query parameter required"}, status_code=400)

    app = get_mcp_workflow()
    inputs = make_initial_state(query)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    async def gen():
        try:
            async for ev in iter_workflow_dashboard_events(app, inputs, config):
                yield sse_encode(ev)
        except Exception as e:
            yield sse_encode({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@mcp.custom_route("/ami/dashboard/resume/stream", methods=["GET"])
async def ami_dashboard_resume_stream(request: Request) -> StreamingResponse | JSONResponse:
    """SSE after HITL: continue with MMR λ (same thread_id as initial stream)."""
    thread_id = (request.query_params.get("thread_id") or "").strip()
    if not thread_id:
        return JSONResponse({"error": "thread_id parameter required"}, status_code=400)
    try:
        lam = float(request.query_params.get("lambda_value", "0.5"))
    except ValueError:
        return JSONResponse({"error": "invalid lambda_value"}, status_code=400)

    app = get_mcp_workflow()

    async def gen():
        try:
            async for ev in iter_workflow_dashboard_events(
                app, Command(resume=lam), {"configurable": {"thread_id": thread_id}}
            ):
                yield sse_encode(ev)
        except Exception as e:
            yield sse_encode({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@mcp.tool()
async def research_company(query: str, thread_id: str = "default") -> str:
    """
    Performs grounded research on semiconductor/automotive supply chains.
    Uses a closed-world graph and vector database to prevent hallucinations.
    Returns multi-hop relationship paths including entity metadata (headquarters, region)
    when available, so you can look for location data in the response.

    Use a stable ``thread_id`` per conversation so HITL resume matches the same checkpoint.
    """
    app = get_mcp_workflow()
    inputs = make_initial_state(query)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    _mcp_progress(f"\n[MCP] Starting workflow for: {query!r} (thread_id={thread_id!r})")

    if _hitl_enabled():
        result = await app.ainvoke(inputs, config)
        rd = _normalize_state(result)
        if rd.get("__interrupt__"):
            log_workflow_outcome(rd, hitl_interrupted=True)
            _mcp_progress("[MCP] Interrupted — waiting for resume_research(lambda_value=...)\n")
            return _interrupt_response(rd, thread_id)
        log_workflow_outcome(rd, hitl_interrupted=False)
        response = (rd.get("response") or "").strip()
        _mcp_progress(f"[MCP] Done (response length={len(response)} chars)\n")
        return response

    merged_state = await _merge_state_from_astream_events(app, inputs, config)

    response = (merged_state.get("response") or "").strip()
    if not response:
        _mcp_progress("[MCP] No response in stream; completing with ainvoke")
        final = await app.ainvoke(inputs, config)
        final_d = _event_output_to_dict(final) or {}
        merged_state.update(final_d)
        response = (merged_state.get("response") or "").strip()
    log_workflow_outcome(merged_state, hitl_interrupted=False)
    _mcp_progress(f"[MCP] Done (response length={len(response)} chars)\n")
    return response


@mcp.tool()
async def resume_research(thread_id: str, lambda_value: float) -> str:
    """
    Resume after HITL interrupt: pass the same ``thread_id`` and MMR λ in [0.0, 1.0].
    Requires ``AMI_HITL_DIVERSITY=1`` and a prior ``research_company`` call that interrupted.

    Uses ``astream_events`` so stderr shows the same ``Node start:`` lines as the non-HITL path
    (re-fetch, grader, synthesis, …). Do not call ``ainvoke(Command(resume))`` again after —
    that would consume a second resume.
    """
    app = get_mcp_workflow()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    lam = float(lambda_value)
    _mcp_progress(f"\n[MCP] Resuming thread {thread_id!r} with λ={lam}\n")

    merged_state = await _merge_state_from_astream_events(
        app, Command(resume=lam), config
    )

    if merged_state.get("__interrupt__"):
        log_workflow_outcome(merged_state, hitl_interrupted=True)
        return _interrupt_response(merged_state, thread_id)

    log_workflow_outcome(merged_state, hitl_interrupted=False)
    response = (merged_state.get("response") or "").strip()
    if not response:
        _mcp_progress(
            "[MCP] No response in merged stream state after resume (check graph output)\n"
        )
    else:
        _mcp_progress(f"[MCP] Resume done (response length={len(response)} chars)\n")
    return response


if __name__ == "__main__":
    _log_mcp_startup_banner()
    if "--stdio" in sys.argv:
        mcp.run()
    else:
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="sse", host="0.0.0.0", port=port)
