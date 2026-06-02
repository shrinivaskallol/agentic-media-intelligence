"""
Graph routing logic for entity workflow.
Extracted for testability without loading full node dependencies.
"""

from app.state.schema import GraphState

# Normalized synthesis_status values (expand over time: partial_grounding, etc.)
SYNTHESIS_ROUTE_TO_CRITIQUE_STATUSES = frozenset({"insufficient_data", "invalid_structured_output"})
SYNTHESIS_SKIP_FAITHFULNESS_JUDGE_STATUSES = frozenset(
    {"insufficient_data", "no_evidence", "invalid_structured_output"}
)
SYNTHESIS_RETRIEVAL_COVERAGE_GAP_STATUSES = frozenset({"insufficient_data", "no_evidence"})
_LLM_SYNTHESIS_STATUSES = frozenset({"complete", "insufficient_data"})


def synthesis_norm_status(state: GraphState | dict) -> str:
    """Lowercase synthesis_status from graph state (empty string if unset)."""
    status = getattr(state, "synthesis_status", None)
    if status is None and isinstance(state, dict):
        status = state.get("synthesis_status", "")
    return str(status).strip().lower()


def synthesis_routes_to_critique(state: GraphState | dict) -> bool:
    """True when the graph should run Critique before Evaluator."""
    if synthesis_norm_status(state) in SYNTHESIS_ROUTE_TO_CRITIQUE_STATUSES:
        return True
    flag = getattr(state, "synthesis_requires_critique", None)
    if flag is None and isinstance(state, dict):
        flag = state.get("synthesis_requires_critique", False)
    return bool(flag)


def synthesis_skip_faithfulness_judge(state: GraphState | dict) -> bool:
    """True when RAGAS/faithfulness judge should be skipped (data gap or broken structured output)."""
    return synthesis_norm_status(state) in SYNTHESIS_SKIP_FAITHFULNESS_JUDGE_STATUSES


def synthesis_retrieval_coverage_gap(state: GraphState | dict) -> bool:
    """True when fallback copy should describe missing/no retrieval vs synthesis-quality failure."""
    return synthesis_norm_status(state) in SYNTHESIS_RETRIEVAL_COVERAGE_GAP_STATUSES


def normalize_llm_synthesis_status(raw: object) -> str | None:
    """Return canonical LLM status or None if missing/invalid (never default to 'complete' here)."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s in _LLM_SYNTHESIS_STATUSES:
        return s
    return None


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
