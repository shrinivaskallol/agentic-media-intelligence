"""
Test route_after_eval routing logic (no heavy imports).
Uses app.utils.routing and app.state.schema only.
"""

import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.state.schema import GraphState
from app.utils.routing import route_after_eval


def test_route_after_eval_returns_critique_when_score_05() -> None:
    """When eval_score is 0.5 (< 0.8), route_after_eval returns 'critique'."""
    state = {"eval_score": 0.5}
    assert route_after_eval(state) == "critique"


def test_route_after_eval_returns_critique_when_score_079() -> None:
    """When eval_score is 0.79 (< 0.8), route_after_eval returns 'critique'."""
    state = {"eval_score": 0.79}
    assert route_after_eval(state) == "critique"


def test_route_after_eval_returns_finish_when_score_10() -> None:
    """When eval_score is 1.0 (>= 0.8), route_after_eval returns 'finish' (maps to END)."""
    state = {"eval_score": 1.0}
    assert route_after_eval(state) == "finish"


def test_route_after_eval_returns_finish_when_score_08() -> None:
    """When eval_score is 0.8 (>= 0.8), route_after_eval returns 'finish'."""
    state = {"eval_score": 0.8}
    assert route_after_eval(state) == "finish"


def test_route_after_eval_with_graph_state() -> None:
    """route_after_eval works with GraphState schema."""
    state = GraphState(query="test", eval_score=0.5)
    assert route_after_eval(state) == "critique"

    state_ok = GraphState(query="test", eval_score=1.0)
    assert route_after_eval(state_ok) == "finish"


def test_route_after_eval_handles_missing_eval_score() -> None:
    """When eval_score is missing, defaults to 0.0 -> critique."""
    state = {}
    assert route_after_eval(state) == "critique"


def test_route_after_eval_uses_ragas_scores_is_passing() -> None:
    """When ragas_scores is present, use is_passing instead of eval_score."""
    assert route_after_eval({"ragas_scores": {"is_passing": True}}) == "finish"
    assert route_after_eval({"ragas_scores": {"is_passing": False}}) == "critique"
