"""
Unit tests for out-of-scope refusal (no retrieval) and grader max-retries → partial synthesis.

Smoke tests (live LLM + DBs):
  uv run ami-workflow "What is the best recipe for sourdough?"
  uv run ami-workflow "Find the 7nm chip provider for a company that doesn't exist"
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from app.logic.workflow import make_initial_state


@pytest.mark.asyncio
async def test_out_of_scope_never_calls_retrieve():
    """Guided refusal path: extractor sets exit_reason; retrieve must not run."""

    def fake_extractor(state):
        return {"exit_reason": "out_of_scope", "entities": [], "intent": "RESEARCH"}

    def fake_refusal(state):
        return {"response": "Guided refusal — pivot to semiconductors."}

    with (
        patch("app.graph.entity_workflow.entity_extractor", fake_extractor),
        patch("app.graph.entity_workflow.refusal_node", fake_refusal),
    ):
        from app.graph.entity_workflow import build_workflow

        app = build_workflow()
        inputs = make_initial_state("What is the best recipe for sourdough?")
        nodes: list[str] = []
        async for chunk in app.astream(inputs):
            for name in chunk.keys():
                nodes.append(name)

        final = await app.ainvoke(inputs)
        fs = final.model_dump() if hasattr(final, "model_dump") else dict(final or {})

    assert "retrieve" not in nodes
    assert "refusal" in nodes
    assert fs.get("exit_reason") == "out_of_scope"
    assert "Guided refusal" in (fs.get("response") or "")


@pytest.mark.asyncio
async def test_max_retries_partial_answer_three_rewrites():
    """
    Grader stays insufficient until retrieval_revision_count reaches 3, then partial synthesis.
    Expect three rewrite→retrieve cycles before synthesis.
    """

    def fake_extractor(state):
        return {
            "entities": ["NonexistentCo"],
            "intent": "RESEARCH",
        }

    def fake_retrieve(state):
        return {"context": [], "retrieved_ids": []}

    def fake_grader(state):
        d = state if isinstance(state, dict) else state.model_dump()
        rev = int(d.get("retrieval_revision_count", 0))
        if rev >= 3:
            return {
                "context_sufficient": True,
                "partial_answer": True,
                "exit_reason": "max_retries_exceeded",
            }
        return {"context_sufficient": False, "retrieval_revision_count": 1}

    def fake_rewrite(state):
        return {}

    def fake_synthesis(state):
        return {
            "response": (
                "PARTIAL: No Knowledge Graph match for the named company; "
                "best-effort from sparse fragments."
            )
        }

    def fake_evaluator(state):
        return {"eval_score": 0.95, "ragas_scores": {"is_passing": True}}

    with (
        patch("app.graph.entity_workflow.entity_extractor", fake_extractor),
        patch("app.graph.entity_workflow.hybrid_retrieval_node", fake_retrieve),
        patch("app.graph.entity_workflow.grader_node", fake_grader),
        patch("app.graph.entity_workflow.rewrite_node", fake_rewrite),
        patch("app.graph.entity_workflow.synthesis_node", fake_synthesis),
        patch("app.graph.entity_workflow.evaluation_node", fake_evaluator),
    ):
        from app.graph.entity_workflow import build_workflow

        app = build_workflow()
        inputs = make_initial_state(
            "Find the 7nm chip provider for a company that doesn't exist"
        )
        nodes: list[str] = []
        async for chunk in app.astream(inputs):
            for name in chunk.keys():
                nodes.append(name)

        final = await app.ainvoke(inputs)
        fs = final.model_dump() if hasattr(final, "model_dump") else dict(final or {})

    assert nodes.count("rewrite") == 3
    assert "retrieve" in nodes
    assert "synthesis" in nodes
    assert fs.get("exit_reason") == "max_retries_exceeded"
    assert fs.get("partial_answer") is True
    assert int(fs.get("retrieval_revision_count", 0)) == 3
    assert "PARTIAL" in (fs.get("response") or "")
