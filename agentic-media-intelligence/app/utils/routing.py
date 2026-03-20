"""
Graph routing logic for entity workflow.
Extracted for testability without loading full node dependencies.
"""

from app.state.schema import GraphState


def route_after_eval(state: GraphState | dict) -> str:
    """
    Route based on RagasMetrics: use ragas_scores['is_passing'] when available.
    Otherwise fall back to eval_score (faithfulness) < 0.8.
    Returns 'critique' (path to critique_node) or 'finish' (END).
    """
    ragas = getattr(state, "ragas_scores", None) or (
        state.get("ragas_scores") if isinstance(state, dict) else None
    )
    if ragas is not None:
        is_passing = ragas.get("is_passing") is True
        if is_passing:
            return "finish"
        return "critique"

    # Fallback when ragas_scores not set (e.g. skip paths)
    score = getattr(state, "eval_score", None)
    if score is None and isinstance(state, dict):
        score = state.get("eval_score", 0.0)
    score = float(score) if score is not None else 0.0
    if score < 0.8:
        return "critique"
    return "finish"
