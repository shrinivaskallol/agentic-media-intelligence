"""
Human-in-the-loop gate: optional interrupt when retrieved context is large (redundancy risk).
Requires AMI_HITL_DIVERSITY=1 and a graph checkpointer. Resumes with MMR λ via Command(resume=...).
"""

from __future__ import annotations

import logging
import os

from langgraph.types import interrupt

from app.state.schema import GraphState

logger = logging.getLogger(__name__)


def _hitl_enabled() -> bool:
    return os.environ.get("AMI_HITL_DIVERSITY", "").strip().lower() in ("1", "true", "yes")


def _min_context_chunks() -> int:
    try:
        return max(1, int(os.environ.get("AMI_HITL_MIN_CONTEXT", "6")))
    except ValueError:
        return 6


def diversity_gate_node(state: GraphState) -> dict:
    """
    If HITL is enabled and enough context chunks were retrieved, pause for human MMR λ.
    On resume, ``interrupt()`` returns the value passed to ``Command(resume=...)``.
    After resume, sets pending_mmr_refetch so the graph loops to ``retrieve`` with the new λ.
    """
    if not _hitl_enabled():
        return {}

    if getattr(state, "mmr_hitl_done", False):
        return {}

    ctx = list(state.context or [])
    threshold = _min_context_chunks()
    if len(ctx) < threshold:
        return {}

    payload = {
        "action": "set_mmr_lambda",
        "message": "Multiple retrieved sources; set diversity (MMR) λ in [0.0, 1.0].",
        "current_context_chunks": len(ctx),
        "suggested_lambda": 0.5,
    }
    logger.info("HITL: interrupting for MMR λ (chunks=%d >= %d)", len(ctx), threshold)

    user_lambda = interrupt(payload)
    try:
        lam = float(user_lambda)
    except (TypeError, ValueError):
        lam = 0.5
    lam = max(0.0, min(1.0, lam))
    return {
        "mmr_lambda": lam,
        "pending_mmr_refetch": True,
        "mmr_hitl_done": True,
    }
