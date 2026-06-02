"""
Human review node: deterministic escalation handoff (not LLM-driven).
"""

import logging

from app.policies.escalation import detect_human_request, evaluate_escalation
from app.state.schema import GraphState

logger = logging.getLogger(__name__)

_REASON_COPY = {
    "user_requested_human": "You asked to speak with a human reviewer.",
    "max_refinement_cycles_exceeded": (
        "The automated quality loop reached the maximum number of revision attempts."
    ),
    "retrieval_exhausted_no_evidence": (
        "Retrieval was retried repeatedly but no grounded evidence was found in the knowledge base."
    ),
    "structured_output_failure": (
        "The synthesis step could not produce valid structured output after multiple attempts."
    ),
    "policy_gap_insufficient_grounding": (
        "Evidence was retrieved, but the system could not produce a policy-compliant grounded answer."
    ),
    "infrastructure_failure_max_revisions": (
        "An upstream service error persisted after the maximum number of quality revision attempts."
    ),
    "infrastructure_failure_retrieval_exhausted": (
        "Retrieval backends reported failures through repeated rewrite cycles; automated recovery stopped."
    ),
}


def human_review_node(state: GraphState | dict) -> dict:
    """Return a fixed escalation message; telemetry uses requires_human_review + escalation_reason."""
    reason = getattr(state, "escalation_reason", None) or (
        state.get("escalation_reason", "") if isinstance(state, dict) else ""
    )
    reason = str(reason or "").strip()
    if not reason:
        query = getattr(state, "query", None) or (
            state.get("query", "") if isinstance(state, dict) else ""
        )
        if detect_human_request(str(query or "")):
            reason = "user_requested_human"
        else:
            _, reason = evaluate_escalation(state)
    reason = reason or "escalation_required"
    detail = _REASON_COPY.get(reason, "This case requires human review.")

    logger.info("HUMAN REVIEW: escalation_reason=%s", reason)

    return {
        "response": (
            "## Human review requested\n\n"
            f"{detail}\n\n"
            "A market-intelligence analyst should review this thread before the answer is "
            "treated as final. Reference code: "
            f"`{reason}`."
        ),
        "requires_human_review": True,
        "is_refused": True,
        "exit_reason": "requires_human_review",
        "escalation_reason": reason,
    }
