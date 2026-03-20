"""
Workflow entry point for programmatic use (e.g. test scripts, MCP).
Provides run_workflow() for sync invocation and run_workflow_async() for async.
"""

import asyncio
from typing import Any

from app.graph.entity_workflow import build_workflow


def make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow (GraphState)."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
    }


async def run_workflow_async(query: str, config: dict | None = None) -> dict[str, Any]:
    """
    Run the entity workflow asynchronously for the given query.
    Returns the final state as a dict.
    """
    app = build_workflow()
    inputs = make_initial_state(query)
    cfg = config or {}
    final_state = await app.ainvoke(inputs, cfg)
    if hasattr(final_state, "model_dump"):
        return final_state.model_dump()
    return dict(final_state) if final_state else {}


def run_workflow(query: str, config: dict | None = None) -> dict[str, Any]:
    """
    Run the entity workflow synchronously for the given query.
    Blocks until completion. Returns the final state as a dict.
    """
    return asyncio.run(run_workflow_async(query, config))
