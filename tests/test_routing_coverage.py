#!/usr/bin/env python3
"""
Routing coverage tests for the entity workflow.
Verifies extractor → router_logic short-circuits on IRRELEVANT and runs full pipeline on RESEARCH/COMPETITION.
Uses async API (astream/ainvoke) because the evaluator node is async.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

# Load .env before app imports (LLM needs API keys)
from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from app.graph.entity_workflow import build_workflow

# Define the test matrix
TEST_CASES = [
    {
        "name": "RESEARCH_INTENT",
        "query": "What is the current state of Nvidia's partnership with TSMC?",
        "expected_nodes": ["extractor", "retrieve", "synthesis"],
        "forbidden_nodes": [],
        "expected_intent": "RESEARCH",
    },
    {
        "name": "COMPETITION_INTENT",
        "query": "How does the Apple M4 chip compare to Nvidia's Blackwell in AI performance?",
        "expected_nodes": ["extractor", "retrieve", "synthesis"],
        "forbidden_nodes": [],
        "expected_intent": "COMPETITION",
    },
    {
        "name": "IRRELEVANT_INTENT_GENERAL",
        "query": "How do I bake a sourdough bread at home?",
        "expected_nodes": ["extractor"],
        "forbidden_nodes": ["retrieve", "synthesis"],
        "expected_intent": "IRRELEVANT",
    },
    {
        "name": "IRRELEVANT_INTENT_WEATHER",
        "query": "What is the weather like in Sunnyvale today?",
        "expected_nodes": ["extractor"],
        "forbidden_nodes": ["retrieve", "synthesis"],
        "expected_intent": "IRRELEVANT",
    },
]


def _make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow (GraphState)."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
        "retrieved_ids": [],
    }


async def run_coverage_suite():
    print("--- STARTING ROUTING COVERAGE TEST ---")
    app = build_workflow()
    passed_count = 0

    for case in TEST_CASES:
        print(f"\n[CASE: {case['name']}]")
        print(f"Query: {case['query']}")

        inputs = _make_initial_state(case["query"])
        actual_nodes = []

        try:
            async for output in app.astream(inputs):
                for node_name in output.keys():
                    actual_nodes.append(node_name)
                    print(f"  -> Executed: {node_name}")

            # Validation: expected nodes present
            passed = True
            for node in case["expected_nodes"]:
                if node not in actual_nodes:
                    print(f"  FAILED: Missing expected node '{node}'. Got {actual_nodes}")
                    passed = False
                    break

            # Validation: forbidden nodes must NOT run (short-circuit check)
            if passed and case.get("forbidden_nodes"):
                hit_forbidden = [n for n in case["forbidden_nodes"] if n in actual_nodes]
                if hit_forbidden:
                    print(f"  FAILED: Executed forbidden nodes {hit_forbidden}. Got {actual_nodes}")
                    passed = False

            if passed:
                print("  RESULT: PASS")
                passed_count += 1
            else:
                print("  RESULT: FAIL")

        except Exception as e:
            print(f"  CRITICAL ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n--- COVERAGE TEST COMPLETE: {passed_count}/{len(TEST_CASES)} PASSED ---")

    # --- FINAL AGENT TRANSCRIPT (Glass Box) ---
    # Run ainvoke on first RESEARCH case to dump full state
    first_research = next(
        (c for c in TEST_CASES if "RESEARCH" in c.get("expected_intent", "")), None
    )
    if first_research and passed_count > 0:
        print("\n=== FINAL AGENT TRANSCRIPT (first RESEARCH case) ===")
        try:
            inputs = _make_initial_state(first_research["query"])
            raw = await app.ainvoke(inputs)
            final_state = (
                raw.model_dump() if hasattr(raw, "model_dump") else (dict(raw) if raw else {})
            )
            print(f"QUERY: {final_state.get('query', '')}")
            print(f"ENTITIES: {final_state.get('entities', [])}")
            print(f"EVIDENCE (RETRIEVED_IDS): {final_state.get('retrieved_ids', [])}")
            print(f"TOTAL REVISIONS: {final_state.get('revision_count', 0)}")
            report = final_state.get("response", "")
            print(f"FINAL REPORT:\n{report[:800]}{'...' if len(report) > 800 else ''}")
            print(f"LAST CRITIQUE: {str(final_state.get('critique', ''))[:300]}...")
            print("=" * 50)
        except Exception as e:
            print(f"[Transcript dump failed: {e}]")

    await asyncio.sleep(0.2)  # Let GenAI client cleanup finish before exit
    return passed_count == len(TEST_CASES)


if __name__ == "__main__":
    try:
        success = asyncio.run(run_coverage_suite())
    except KeyboardInterrupt:
        success = False
    sys.exit(0 if success else 1)
