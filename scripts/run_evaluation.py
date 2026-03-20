#!/usr/bin/env python3
"""
Run automated evaluation against golden dataset.
Uses Judge LLM to score faithfulness/relevance. Saves failures to tests/eval_failures.json.
Run: uv run python scripts/run_evaluation.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph.entity_workflow import build_workflow
from app.llm_factory import get_llm
from app.prompts import get_prompt

# ---------------------------------------------------------------------------
# Judge Schema
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Structured output from the Judge LLM for non-trap cases."""

    faithfulness: float = Field(
        ge=0,
        le=1,
        description="0-1: Does the answer contradict the ground truth? 1 = fully consistent.",
    )
    relevance: float = Field(
        ge=0,
        le=1,
        description="0-1: Does it answer the user's specific query? 1 = fully relevant.",
    )
    reasoning: str = Field(
        default="",
        description="Brief justification for the scores.",
    )


class TrapVerdict(BaseModel):
    """Structured output for trap cases: did the agent refuse appropriately?"""

    refusal_correct: bool = Field(
        description="True if the agent refused, stated data is insufficient, or did not hallucinate.",
    )
    reasoning: str = Field(
        default="",
        description="Brief justification.",
    )


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------


def _make_initial_state(query: str) -> dict:
    """Create initial state for the entity workflow."""
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
    }


def _is_refusal(response: str) -> bool:
    """Check if the agent's response indicates a refusal (insufficient data, etc.)."""
    if not response or not response.strip():
        return True
    r = response.upper()
    return (
        "REFUSAL" in r
        or "INSUFFICIENT" in r
        or "NOT IN" in r
        or "NO DATA" in r
        or "DO NOT HAVE" in r
        or "CANNOT FIND" in r
        or "OUTSIDE MY" in r
        or "BEYOND MY" in r
    )


async def _judge_non_trap(
    query: str,
    agent_response: str,
    ground_truth_fact: str,
    llm,
) -> JudgeVerdict:
    """Use Judge LLM to score faithfulness and relevance."""
    schema_llm = llm.with_structured_output(JudgeVerdict)
    prompt = get_prompt(
        "evaluation",
        "judge_non_trap",
        query=query,
        ground_truth_fact=ground_truth_fact,
        agent_response=agent_response,
    )
    messages = [HumanMessage(content=prompt)]
    return await schema_llm.ainvoke(messages)


async def _judge_trap(
    query: str,
    agent_response: str,
    llm,
) -> TrapVerdict:
    """Check if the agent correctly refused for an out-of-scope question."""
    schema_llm = llm.with_structured_output(TrapVerdict)
    prompt = get_prompt(
        "evaluation",
        "judge_trap",
        query=query,
        agent_response=agent_response,
    )
    messages = [HumanMessage(content=prompt)]
    return await schema_llm.ainvoke(messages)


async def run_single(
    case: dict,
    app,
    judge_llm,
) -> dict:
    """Run one test case and return result dict."""
    cid = case.get("id", "unknown")
    query = case["query"]
    ground_truth = case.get("ground_truth_fact", "")
    category = case.get("category", "")
    expect_refusal = case.get("expect_refusal", False)

    inputs = _make_initial_state(query)
    start = time.perf_counter()
    try:
        final_state = await app.ainvoke(inputs)
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "id": cid,
            "query": query,
            "category": category,
            "expect_refusal": expect_refusal,
            "status": "error",
            "error": str(e),
            "latency_s": elapsed,
            "faithfulness": None,
            "relevance": None,
            "refusal_correct": None,
            "agent_response": "",
        }

    elapsed = time.perf_counter() - start
    response = final_state.get("response", "")

    result = {
        "id": cid,
        "query": query,
        "category": category,
        "expect_refusal": expect_refusal,
        "status": "ok",
        "latency_s": round(elapsed, 3),
        "agent_response": response[:500] + ("..." if len(response) > 500 else ""),
    }

    if expect_refusal:
        verdict = await _judge_trap(query, response, judge_llm)
        result["refusal_correct"] = verdict.refusal_correct
        result["faithfulness"] = None
        result["relevance"] = None
        result["judge_reasoning"] = verdict.reasoning
    else:
        verdict = await _judge_non_trap(query, response, ground_truth, judge_llm)
        result["faithfulness"] = round(verdict.faithfulness, 2)
        result["relevance"] = round(verdict.relevance, 2)
        result["refusal_correct"] = None
        result["judge_reasoning"] = verdict.reasoning

    return result


def _passed(result: dict) -> bool:
    """Determine if a result is a pass."""
    if result.get("status") == "error":
        return False
    if result.get("expect_refusal"):
        return result.get("refusal_correct") is True
    f = result.get("faithfulness")
    r = result.get("relevance")
    return f is not None and r is not None and f >= 0.7 and r >= 0.7


async def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    dataset_path = _proj / "tests" / "eval_dataset.json"
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)

    with open(dataset_path) as f:
        cases = json.load(f)

    console.print(f"\n[bold]Evaluating {len(cases)} test cases[/bold]\n")

    app = build_workflow()
    judge_llm = get_llm()

    results = []
    for i, case in enumerate(cases):
        console.print(f"  [{i + 1}/{len(cases)}] {case['id']} ... ", end="")
        r = await run_single(case, app, judge_llm)
        results.append(r)
        if r.get("status") == "error":
            console.print("[red]ERROR[/red]")
        elif r.get("expect_refusal"):
            console.print("[green]PASS[/green]" if r.get("refusal_correct") else "[red]FAIL[/red]")
        else:
            f, rel = r.get("faithfulness"), r.get("relevance")
            ok = f is not None and rel is not None and f >= 0.7 and rel >= 0.7
            console.print(
                f"[green]PASS[/green] (F={f} R={rel})" if ok else f"[red]FAIL[/red] (F={f} R={rel})"
            )

    # Summary table
    table = Table(title="Evaluation Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Category")
    table.add_column("Query", max_width=45, overflow="fold")
    table.add_column("Latency (s)")
    table.add_column("Faithfulness")
    table.add_column("Relevance")
    table.add_column("Refusal OK")
    table.add_column("Pass", style="bold")

    for r in results:
        passed = _passed(r)
        table.add_row(
            r["id"],
            r.get("category", ""),
            r["query"][:45] + ("..." if len(r["query"]) > 45 else ""),
            str(r.get("latency_s", "-")),
            str(r.get("faithfulness", "-")) if r.get("faithfulness") is not None else "-",
            str(r.get("relevance", "-")) if r.get("relevance") is not None else "-",
            str(r.get("refusal_correct")) if r.get("refusal_correct") is not None else "-",
            "[green]YES[/green]" if passed else "[red]NO[/red]",
        )

    console.print()
    console.print(table)

    passed_count = sum(1 for r in results if _passed(r))
    avg_latency = sum(r.get("latency_s", 0) for r in results if r.get("status") == "ok") / max(
        1, sum(1 for r in results if r.get("status") == "ok")
    )
    console.print(f"\n[bold]Total: {passed_count}/{len(results)} passed[/bold]")
    console.print(f"Avg latency: {avg_latency:.2f}s\n")

    # Save failures
    failures = [
        {
            "id": r["id"],
            "query": r["query"],
            "category": r.get("category"),
            "expect_refusal": r.get("expect_refusal"),
            "status": r.get("status"),
            "error": r.get("error"),
            "agent_response": r.get("agent_response"),
            "faithfulness": r.get("faithfulness"),
            "relevance": r.get("relevance"),
            "refusal_correct": r.get("refusal_correct"),
            "judge_reasoning": r.get("judge_reasoning"),
        }
        for r in results
        if not _passed(r)
    ]

    report_path = _proj / "tests" / "eval_failures.json"
    with open(report_path, "w") as f:
        json.dump(failures, f, indent=2)

    console.print(f"Failures saved to: [cyan]{report_path}[/cyan]")
    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
