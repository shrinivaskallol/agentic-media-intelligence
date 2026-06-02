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
the graph interrupts; call ``resume_research`` with the same ``thread_id`` and ``lambda_value``
after ``run_market_research`` returns an interrupt payload.

Dashboard SSE (Streamlit): ``GET /ami/dashboard/stream?query=...&thread_id=...`` and
``GET /ami/dashboard/resume/stream?thread_id=...&lambda_value=...`` — custom HTTP routes on the
same port as MCP SSE (e.g. :8000).
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

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
from pydantic import Field
from langgraph.types import Command
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from app.errors.helpers import (
    classify_exception,
    finalize_mcp_workflow_response,
    format_mcp_error,
    validation_error,
)
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
        "HITL ON — run_market_research uses ainvoke until interrupt; resume_research streams"
        if hitl
        else "HITL OFF — run_market_research streams to completion (no interrupt pause)"
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


@mcp.custom_route("/ami/dashboard/stream", methods=["GET"])
async def ami_dashboard_workflow_stream(request: Request) -> StreamingResponse | JSONResponse:
    """SSE of LangGraph node/tool events + final state (for Streamlit or other clients)."""
    query = (request.query_params.get("query") or "").strip()
    thread_id = (request.query_params.get("thread_id") or "dashboard").strip()
    if not query:
        return JSONResponse(
            validation_error("query parameter required", field="query").model_dump(),
            status_code=400,
        )

    app = get_mcp_workflow()
    inputs = make_initial_state(query)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    async def gen():
        try:
            async for ev in iter_workflow_dashboard_events(app, inputs, config):
                yield sse_encode(ev)
        except Exception as e:
            yield sse_encode(
                {
                    "type": "error",
                    "error": classify_exception(e, component="dashboard", operation="stream").model_dump(),
                }
            )

    return StreamingResponse(gen(), media_type="text/event-stream")


@mcp.custom_route("/ami/dashboard/resume/stream", methods=["GET"])
async def ami_dashboard_resume_stream(request: Request) -> StreamingResponse | JSONResponse:
    """SSE after HITL: continue with MMR λ (same thread_id as initial stream)."""
    thread_id = (request.query_params.get("thread_id") or "").strip()
    if not thread_id:
        return JSONResponse(
            validation_error("thread_id parameter required", field="thread_id").model_dump(),
            status_code=400,
        )
    try:
        lam = float(request.query_params.get("lambda_value", "0.5"))
    except ValueError:
        return JSONResponse(
            validation_error("invalid lambda_value", field="lambda_value").model_dump(),
            status_code=400,
        )

    app = get_mcp_workflow()

    async def gen():
        try:
            async for ev in iter_workflow_dashboard_events(
                app, Command(resume=lam), {"configurable": {"thread_id": thread_id}}
            ):
                yield sse_encode(ev)
        except Exception as e:
            yield sse_encode(
                {
                    "type": "error",
                    "error": classify_exception(
                        e, component="dashboard", operation="resume_stream"
                    ).model_dump(),
                }
            )

    return StreamingResponse(gen(), media_type="text/event-stream")


@mcp.tool()
async def run_market_research(
    query: Annotated[
        str,
        Field(
            description=(
                "Natural-language market-intelligence question: company research, competitor "
                "analysis, semiconductor or automotive supply chains, products, or market dynamics. "
                "Not for general chat, weather, or unrelated topics."
            ),
        ),
    ],
    thread_id: Annotated[
        str,
        Field(
            description=(
                "Stable conversation identifier for LangGraph checkpointing. Reuse the same value "
                "for follow-ups in one session and for resume_research after a HITL interrupt."
            ),
        ),
    ] = "default",
) -> str:
    """
    Run the full AMI workflow: extract entities → retrieve (graph + vector) → grade → synthesize.

    WHEN TO USE:
    - User wants a grounded report from the knowledge graph and vector store.
    - First call in a thread, or any new question that should run the full pipeline.

    WHEN NOT TO USE:
    - After a HITL interrupt JSON was returned (use resume_research with the same thread_id).
    - Database health checks (not exposed here).

    RETURNS (always ``str``):
    - **Normal completion:** Markdown intelligence report (may include entity/location metadata
      from graph paths). Not JSON on success.
    - **HITL pause** (only if ``AMI_HITL_DIVERSITY=1``): JSON string with
      ``{"status":"interrupted","thread_id":...,"interrupt":[...],"hint":...}``.
      This is a pause, not an error — call resume_research next.
    - **Error JSON:** ``{"status":"error","error":{ToolError...}}`` on infrastructure/operation failure.
    - **Empty-result JSON:** ``{"status":"empty_result","empty_result":{...}}`` when policy reports no evidence
      (not an error — distinct from service failure).
    - Never returns a bare empty string on operational failure.

    Does not raise MCP exceptions for out-of-scope queries; those are handled inside the workflow text.
    """
    if not (query or "").strip():
        return format_mcp_error(
            validation_error("query must be a non-empty string", field="query"),
            thread_id=thread_id,
        )

    app = get_mcp_workflow()
    inputs = make_initial_state(query.strip())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    _mcp_progress(f"\n[MCP] Starting workflow for: {query!r} (thread_id={thread_id!r})")

    try:
        if _hitl_enabled():
            result = await app.ainvoke(inputs, config)
            merged_state = _normalize_state(result)
        else:
            merged_state = await _merge_state_from_astream_events(app, inputs, config)
            if not (merged_state.get("response") or "").strip():
                _mcp_progress("[MCP] No response in stream; completing with ainvoke")
                final = await app.ainvoke(inputs, config)
                final_d = _event_output_to_dict(final) or {}
                merged_state.update(final_d)

        hitl = bool(merged_state.get("__interrupt__"))
        log_workflow_outcome(merged_state, hitl_interrupted=hitl)
        if hitl:
            _mcp_progress("[MCP] Interrupted — waiting for resume_research(lambda_value=...)\n")
        else:
            out = finalize_mcp_workflow_response(merged_state, thread_id)
            _mcp_progress(f"[MCP] Done (return length={len(out)} chars)\n")
        return finalize_mcp_workflow_response(merged_state, thread_id)
    except Exception as e:
        logger_exc = classify_exception(e, component="mcp", operation="run_market_research")
        _mcp_progress(f"[MCP] Error: {logger_exc.message}\n")
        return format_mcp_error(logger_exc, thread_id=thread_id)


@mcp.tool()
async def resume_research(
    thread_id: Annotated[
        str,
        Field(
            description=(
                "Same thread_id passed to run_market_research when it returned "
                'JSON with status="interrupted".'
            ),
        ),
    ],
    lambda_value: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "MMR diversity weight for re-ranking retrieved chunks: 1.0 = relevance-only; "
                "0.0 = maximum diversity among candidates."
            ),
        ),
    ],
) -> str:
    """
    Continue a paused workflow after human-in-the-loop MMR selection.

    WHEN TO USE:
    - Prior run_market_research returned JSON with ``status`` = ``"interrupted"``.
    - Server has ``AMI_HITL_DIVERSITY`` enabled.

    WHEN NOT TO USE:
    - New user question (use run_market_research).
    - HITL disabled on the server (no matching checkpoint interrupt).

    RETURNS (always ``str``):
    - Same contract as run_market_research (Markdown, interrupt JSON, error JSON, or empty_result JSON).
    """
    if not (thread_id or "").strip():
        return format_mcp_error(
            validation_error("thread_id must be a non-empty string", field="thread_id"),
        )

    app = get_mcp_workflow()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    lam = float(lambda_value)
    _mcp_progress(f"\n[MCP] Resuming thread {thread_id!r} with λ={lam}\n")

    try:
        merged_state = await _merge_state_from_astream_events(
            app, Command(resume=lam), config
        )
        hitl = bool(merged_state.get("__interrupt__"))
        log_workflow_outcome(merged_state, hitl_interrupted=hitl)
        out = finalize_mcp_workflow_response(merged_state, thread_id)
        _mcp_progress(f"[MCP] Resume done (return length={len(out)} chars)\n")
        return out
    except Exception as e:
        err = classify_exception(e, component="mcp", operation="resume_research")
        _mcp_progress(f"[MCP] Resume error: {err.message}\n")
        return format_mcp_error(err, thread_id=thread_id)


if __name__ == "__main__":
    _log_mcp_startup_banner()
    if "--stdio" in sys.argv:
        mcp.run()
    else:
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="sse", host="0.0.0.0", port=port)
