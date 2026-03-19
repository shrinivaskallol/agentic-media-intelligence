#!/usr/bin/env python3
"""
MCP Server: Agent-as-a-Service for the Agentic Research workflow.
Exposes the validated research flow (extract → retrieve → synthesis) as an MCP tool.
Run: uv run python src/mcp_server.py
"""

import os
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
os.chdir(_proj)

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from fastmcp import FastMCP

from app.graph.entity_workflow import build_workflow


mcp = FastMCP("Auto-Intelligence-Service")


def _make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow (GraphState)."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
    }


@mcp.tool()
async def research_company(query: str) -> str:
    """
    Performs grounded research on semiconductor/automotive supply chains.
    Uses a closed-world graph and vector database to prevent hallucinations.
    Returns multi-hop relationship paths including entity metadata (headquarters, region)
    when available, so you can look for location data in the response.
    """
    app = build_workflow()
    inputs = _make_initial_state(query)
    # Redirect stdout/stderr during workflow: MCP uses stdio for JSON-RPC.
    # Workflow print() calls would corrupt the stream and cause "Unexpected token" errors.
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = devnull
            result = await app.ainvoke(inputs)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    return result.get("response", "")


if __name__ == "__main__":
    mcp.run()
