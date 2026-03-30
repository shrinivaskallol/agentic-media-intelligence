#!/usr/bin/env python3
"""
End-to-end HITL + MMR smoke test (no MCP).

Requires: .env with DB/LLM as usual. Sets AMI_HITL_* for this process only.

Usage (from repo root):
  AMI_HITL_DIVERSITY=1 AMI_HITL_MIN_CONTEXT=2 uv run python scripts/test_hitl_mmr.py

AMI_HITL_MIN_CONTEXT must be <= len(state.context) after retrieve (currently 2 blocks:
graph section + vector section). Default 6 never fires; use 1 or 2.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
os.chdir(_proj)

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

# Override for this run so HITL always arms (override .env if AMI_HITL_DIVERSITY=0).
os.environ["AMI_HITL_DIVERSITY"] = "1"
os.environ["AMI_HITL_MIN_CONTEXT"] = os.environ.get("AMI_HITL_MIN_CONTEXT", "2")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph.entity_workflow import build_workflow
from app.logic.workflow import make_initial_state


async def main() -> None:
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the current state of Nvidia's partnership with TSMC?"
    )
    thread_id = "hitl-smoke-test"
    lam = 0.25
    if len(sys.argv) > 2:
        lam = float(sys.argv[2])

    app = build_workflow(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = make_initial_state(query)

    print("--- First ainvoke (expect interrupt if threshold met) ---")
    r1 = await app.ainvoke(inputs, config)
    d1 = r1.model_dump() if hasattr(r1, "model_dump") else dict(r1)

    intr = d1.get("__interrupt__")
    if not intr:
        print("No interrupt. context items:", len(d1.get("context") or []))
        print("Set AMI_HITL_MIN_CONTEXT to 1 or 2 (see script docstring).")
        sys.exit(1)

    print("Interrupt payload present:", bool(intr))
    print("--- Resume with λ =", lam, "---")
    r2 = await app.ainvoke(Command(resume=lam), config)
    d2 = r2.model_dump() if hasattr(r2, "model_dump") else dict(r2)

    if d2.get("__interrupt__"):
        print("Still interrupted after resume:", d2["__interrupt__"])
        sys.exit(2)

    print("mmr_lambda:", d2.get("mmr_lambda"))
    print("retrieved_ids (vector):", d2.get("retrieved_ids"))
    print("response preview:", (d2.get("response") or "")[:400])
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
