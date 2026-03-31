"""
Shared LangGraph → JSON events for dashboards (SSE, Streamlit).

Yields small dicts suitable for ``data: ...`` Server-Sent Events lines.
Does not import the MCP server (avoids circular imports).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command


def event_output_to_dict(output: Any) -> dict | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return None


def _serialize_interrupt_state(state: dict[str, Any]) -> dict[str, Any]:
    """Make merged state JSON-safe for SSE (interrupt payloads can be objects)."""
    intr = state.get("__interrupt__") or []
    serial = []
    for it in intr:
        if hasattr(it, "value"):
            v = it.value
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                serial.append({"value": v, "id": getattr(it, "id", None)})
            else:
                serial.append({"value": str(v), "id": getattr(it, "id", None)})
        else:
            serial.append(str(it))
    out = {k: v for k, v in state.items() if k != "__interrupt__"}
    out["__interrupt__"] = serial
    return out


async def iter_workflow_dashboard_events(
    app: Any,
    inputs: dict[str, Any] | Command,
    config: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream ``astream_events`` (v2) as dashboard events.

    Event types: ``node_start``, ``tool_start``, ``interrupt``, ``done``, ``error`` (caller).
    """
    merged_state: dict[str, Any] = {}
    try:
        async for event in app.astream_events(inputs, config, version="v2"):
            kind = event.get("event")
            meta = event.get("metadata") or {}

            if kind == "on_chain_start" and meta.get("langgraph_node"):
                yield {
                    "type": "node_start",
                    "node": meta["langgraph_node"],
                }
            elif kind == "on_tool_start":
                yield {"type": "tool_start", "name": event.get("name", "?")}

            if kind == "on_chain_end":
                data = event.get("data") or {}
                out = event_output_to_dict(data.get("output"))
                if out:
                    merged_state.update(out)
                    if out.get("__interrupt__"):
                        yield {
                            "type": "interrupt",
                            "thread_id": config.get("configurable", {}).get("thread_id"),
                            "state": _serialize_interrupt_state(dict(merged_state)),
                            "hint": "Submit λ and call resume stream with same thread_id.",
                        }
                        yield {
                            "type": "done",
                            "status": "interrupted",
                            "state": _serialize_interrupt_state(dict(merged_state)),
                        }
                        return

        # astream_events often ends without __interrupt__ in on_chain_end when diversity_gate
        # calls interrupt(); use checkpoint snapshot for authoritative values + pending interrupts.
        try:
            snap = await app.aget_state(config)
        except Exception:
            snap = None
        if snap is not None and isinstance(snap.values, dict):
            merged_state.update(snap.values)
        if snap is not None and snap.interrupts:
            yield {
                "type": "interrupt",
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "state": _serialize_interrupt_state(dict(merged_state)),
                "hint": "Graph paused for HITL — submit λ and open resume stream.",
            }
            yield {
                "type": "done",
                "status": "interrupted",
                "state": _serialize_interrupt_state(dict(merged_state)),
            }
            return

        safe_state = json.loads(json.dumps(dict(merged_state), default=str))
        has_resp = bool(str(merged_state.get("response") or "").strip())
        exit_reason = str(merged_state.get("exit_reason") or "").strip()

        if exit_reason == "out_of_scope":
            yield {
                "type": "status",
                "status": "refusal",
                "exit_reason": exit_reason,
            }
        elif exit_reason == "max_retries_exceeded":
            yield {
                "type": "status",
                "status": "partial_warning",
                "exit_reason": exit_reason,
                "message": (
                    "Notice: Data for this specific query is sparse; providing a partial analysis "
                    "based on high-confidence fragments."
                ),
            }

        if exit_reason == "out_of_scope":
            done_status = "refusal"
        elif exit_reason == "max_retries_exceeded":
            done_status = "completed_partial"
        elif has_resp:
            done_status = "completed"
        else:
            done_status = "partial"

        yield {"type": "done", "status": done_status, "state": safe_state}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        raise


def sse_encode(obj: dict[str, Any]) -> bytes:
    """One SSE message (UTF-8)."""
    return f"data: {json.dumps(obj, default=str)}\n\n".encode()
