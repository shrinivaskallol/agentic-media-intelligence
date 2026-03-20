#!/usr/bin/env python3
"""
Test persistent memory via SQLite checkpointer.
Verifies that state persists across invocations when using the same thread_id.
If the second query correctly identifies "their" as a reference to the company
from the first query (Nvidia), the system is retrieving state from the checkpoint DB.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.entity_workflow import build_workflow, get_checkpoint_db_path


def _make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow (GraphState)."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
    }


async def run_test():
    db_path = get_checkpoint_db_path()
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        app = build_workflow(checkpointer=saver)
        config = {"configurable": {"thread_id": "principal_test_001"}}

        # Step 1: Establish context about Nvidia
        print("--- Step 1: Establishing context ---")
        inputs1 = _make_initial_state("Tell me about Nvidia.")
        try:
            await app.ainvoke(inputs1, config)
        finally:
            await asyncio.sleep(0.2)  # Let httpx/Groq collectors finish before event loop closes
        print("First invocation complete.\n")

        # Step 2: Test memory with an anaphoric reference ("their")
        # Pass ONLY the new query so checkpoint state (entities, history) is preserved for coreference
        print("--- TESTING PERSISTENT MEMORY ---")
        print("Query: 'What is their latest chip architecture?' (their -> Nvidia)\n")
        inputs2 = {"query": "What is their latest chip architecture?"}

        report = None
        try:
            async for output in app.astream(inputs2, config):
                for node, data in output.items():
                    if node == "synthesis":
                        report = data.get("response", "")
                    elif node == "extractor" and data.get("response"):
                        report = data["response"]
        finally:
            await asyncio.sleep(0.2)  # Let httpx/Groq collectors finish before event loop closes

        if report:
            print("--- Report ---")
            print(report)
            # Success heuristic: report mentions Nvidia (anaphora resolved)
            if "nvidia" in report.lower():
                print(
                    "\n[PASS] Report references Nvidia—anaphoric 'their' was resolved from checkpoint memory."
                )
            else:
                print("\n[INFO] Report generated. Check if 'their' was resolved to Nvidia.")
        else:
            print("[FAIL] No report generated.")


if __name__ == "__main__":
    asyncio.run(run_test())
