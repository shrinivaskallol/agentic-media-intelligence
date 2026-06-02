"""
Fallback node: Evidence-only presentation when synthesis fails or refuses.
Tracks retrieval coverage gaps (insufficient_data, no_evidence) vs synthesis-quality failure for telemetry.
"""

from app.state.schema import GraphState
from app.utils.routing import synthesis_retrieval_coverage_gap


def fallback_node(state: GraphState | dict) -> dict:
    """
    SOTA fallback: when the agent refuses (insufficient data) or fails quality after 3 revisions,
    return a professional message plus raw evidence. Sets is_refused=True for telemetry.
    """
    ctx = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    context_str = "\n\n".join(ctx) if ctx else "(No context retrieved.)"

    is_coverage_gap = synthesis_retrieval_coverage_gap(state)

    report = (
        "CONFIDENCE ALERT: The system could not find enough relevant data to answer "
        "your specific query with high precision."
    )
    if is_coverage_gap:
        report += "\n\nREASON: No relevant documents found in the current intelligence database."
    else:
        report += (
            "\n\nREASON: Synthesis failed to meet accuracy thresholds after multiple attempts."
        )

    report += f"\n\nRAW EVIDENCE RETRIEVED:\n{context_str}"

    return {
        "response": report,
        "is_refused": True,
        "critique_score": 0.0,
    }
