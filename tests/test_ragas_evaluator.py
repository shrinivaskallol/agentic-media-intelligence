"""
Test evaluator node (Judge with RagasMetrics).
Routing tests are in test_ragas_routing.py (avoids heavy imports).

Mocks get_critique_llm and the Judge chain to return RagasMetrics.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.nodes.evaluator import RagasMetrics, evaluation_node
from app.state.schema import GraphState

# ---------------------------------------------------------------------------
# Fixtures: GraphState-compatible input states
# ---------------------------------------------------------------------------


@pytest.fixture
def hallucinated_state() -> dict:
    """State with a response that would score low (hallucinated)."""
    return {
        "query": "What process does Blackwell use?",
        "response": "Blackwell uses Intel 7nm and is manufactured by Samsung.",
        "context": [
            "--- GRAPH KNOWLEDGE ---\nGRAPH FACT: Blackwell MANUFACTURED_BY TSMC",
            "--- TEXT CHUNKS ---\nNvidia Blackwell uses TSMC 4NP process.",
        ],
    }


@pytest.fixture
def correct_state() -> dict:
    """State with a response that would score high (grounded)."""
    return {
        "query": "What process does Blackwell use?",
        "response": "Blackwell is manufactured by TSMC using the 4NP process.",
        "context": [
            "--- GRAPH KNOWLEDGE ---\nGRAPH FACT: Blackwell MANUFACTURED_BY TSMC",
            "--- TEXT CHUNKS ---\nNvidia Blackwell uses TSMC 4NP process.",
        ],
    }


@pytest.fixture
def graph_state_hallucinated(hallucinated_state: dict) -> GraphState:
    """GraphState instance for hallucinated case."""
    return GraphState(**hallucinated_state)


@pytest.fixture
def graph_state_correct(correct_state: dict) -> GraphState:
    """GraphState instance for correct case."""
    return GraphState(**correct_state)


# ---------------------------------------------------------------------------
# Tests: evaluation_node
# ---------------------------------------------------------------------------


def _fake_metrics(faithfulness: float, **kwargs) -> RagasMetrics:
    return RagasMetrics(
        faithfulness=faithfulness,
        answer_relevancy=kwargs.get("answer_relevancy", faithfulness),
        context_precision=kwargs.get("context_precision", faithfulness),
        critique_instruction=kwargs.get("critique_instruction", ""),
        is_passing=kwargs.get("is_passing", faithfulness >= 0.8),
    )


@pytest.mark.asyncio
async def test_evaluation_node_returns_05_for_hallucinated(hallucinated_state: dict) -> None:
    """Mock Judge to return faithfulness=0.5; eval_score should be 0.5."""
    fake = _fake_metrics(0.5, is_passing=False, critique_instruction="Fix the hallucination")
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake)
    with patch("app.nodes.evaluator.get_critique_llm") as m:
        m.return_value.with_structured_output.return_value = mock_chain
        result = await evaluation_node(hallucinated_state)
    assert result["eval_score"] == 0.5
    assert result["ragas_scores"]["faithfulness"] == 0.5
    assert result["ragas_scores"]["is_passing"] is False
    assert "critique_instruction" in result


@pytest.mark.asyncio
async def test_evaluation_node_returns_10_for_correct(correct_state: dict) -> None:
    """Mock Judge to return faithfulness=1.0; eval_score should be 1.0."""
    fake = _fake_metrics(1.0, is_passing=True)
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake)
    with patch("app.nodes.evaluator.get_critique_llm") as m:
        m.return_value.with_structured_output.return_value = mock_chain
        result = await evaluation_node(correct_state)
    assert result["eval_score"] == 1.0
    assert result["ragas_scores"]["is_passing"] is True


@pytest.mark.asyncio
async def test_evaluation_node_skips_when_no_context() -> None:
    """When context is empty, evaluation_node returns eval_score=1.0 (pass through)."""
    state = {"query": "test", "response": "some response", "context": []}
    result = await evaluation_node(state)
    assert result["eval_score"] == 1.0
    assert result["ragas_scores"] is None


@pytest.mark.asyncio
async def test_evaluation_node_skips_on_refusal() -> None:
    """When synthesis_status is insufficient_data, skip Judge and set is_refused=True."""
    state = {
        "query": "test",
        "response": "I could not ground this in the retrieved evidence.",
        "context": ["some context"],
        "synthesis_status": "insufficient_data",
    }
    result = await evaluation_node(state)
    assert result["eval_score"] == 1.0
    assert result["is_refused"] is True


@pytest.mark.asyncio
async def test_evaluation_node_skips_on_no_evidence() -> None:
    """no_evidence skips Judge like a data-gap verdict."""
    state = {
        "query": "test",
        "response": "## No evidence",
        "context": ["some context"],
        "synthesis_status": "no_evidence",
    }
    result = await evaluation_node(state)
    assert result["eval_score"] == 1.0
    assert result["is_refused"] is True


@pytest.mark.asyncio
async def test_evaluation_node_skips_on_invalid_structured_output() -> None:
    state = {
        "query": "test",
        "response": "stub",
        "context": ["some context"],
        "synthesis_status": "invalid_structured_output",
    }
    result = await evaluation_node(state)
    assert result["eval_score"] == 1.0
    assert result["is_refused"] is True


@pytest.mark.asyncio
async def test_evaluation_node_uses_graph_state(graph_state_hallucinated: GraphState) -> None:
    """evaluation_node works with GraphState schema."""
    fake = _fake_metrics(0.7, is_passing=False)
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake)
    with patch("app.nodes.evaluator.get_critique_llm") as m:
        m.return_value.with_structured_output.return_value = mock_chain
        result = await evaluation_node(graph_state_hallucinated)
    assert result["eval_score"] == 0.7
    assert result["ragas_scores"]["faithfulness"] == 0.7


@pytest.mark.asyncio
async def test_evaluation_node_sets_is_refused_only_after_critique_attempted() -> None:
    """is_refused is True only when faithfulness high, not passing, AND revision_count >= 1."""
    fake = _fake_metrics(
        faithfulness=0.95,
        answer_relevancy=0.5,
        context_precision=0.9,
        is_passing=False,
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=fake)
    with patch("app.nodes.evaluator.get_critique_llm") as m:
        m.return_value.with_structured_output.return_value = mock_chain
        # revision_count=0: must NOT set is_refused (Critique hasn't run yet)
        result = await evaluation_node(
            {
                "query": "List three partners",
                "response": "TSMC is a partner.",
                "context": ["TSMC is a partner."],
                "revision_count": 0,
            }
        )
    assert result["eval_score"] == 0.95
    assert result["is_refused"] is False  # No critique attempt yet

    # revision_count=1: Critique ran once; now is_refused can be True
    with patch("app.nodes.evaluator.get_critique_llm") as m:
        m.return_value.with_structured_output.return_value = mock_chain
        result = await evaluation_node(
            {
                "query": "List three partners",
                "response": "TSMC is a partner.",
                "context": ["TSMC is a partner."],
                "revision_count": 1,
            }
        )
    assert result["eval_score"] == 0.95
    assert result["is_refused"] is True
