"""Pre-retrieval hooks: budget caps and circuit breaker."""

from __future__ import annotations

import logging
from typing import Any

from app.policies.config import (
    MAX_ENTITIES,
    MAX_GRAPH_LIMIT,
    MAX_RETRIEVAL_REVISIONS,
    MAX_VECTOR_LIMIT,
)

logger = logging.getLogger(__name__)


def _retrieval_revision_count(state: Any) -> int:
    rev = getattr(state, "retrieval_revision_count", None)
    if rev is None and isinstance(state, dict):
        rev = state.get("retrieval_revision_count", 0)
    return int(rev) if rev is not None else 0


def apply_retrieval_budget(
    entities: list[str],
    intent: str,
) -> tuple[list[str], int, int]:
    """
    PreToolUse: cap entities and per-source fetch limits before DB tools run.
    """
    capped = list(entities or [])[:MAX_ENTITIES]
    if len(entities or []) > len(capped):
        logger.info(
            "POLICY PreRetrieve: capped entities %d → %d",
            len(entities),
            len(capped),
        )

    intent_u = (intent or "RESEARCH").strip().upper()
    if intent_u == "COMPETITION":
        graph_limit = min(5, MAX_GRAPH_LIMIT)
        vector_limit = min(3, MAX_VECTOR_LIMIT)
    else:
        graph_limit = MAX_GRAPH_LIMIT
        vector_limit = MAX_VECTOR_LIMIT

    return capped, graph_limit, vector_limit


def check_retrieval_circuit_breaker(state: Any) -> dict | None:
    """
    PreToolUse: block another DB fetch when retrieval revision budget is exhausted.
    Returns a state patch to merge (skip tool execution) or None to proceed.
    """
    rev = _retrieval_revision_count(state)
    if rev < MAX_RETRIEVAL_REVISIONS:
        return None

    logger.warning(
        "POLICY PreRetrieve: circuit breaker — retrieval_revision_count=%s >= %s; skipping fetch",
        rev,
        MAX_RETRIEVAL_REVISIONS,
    )
    ctx = getattr(state, "context", None)
    if ctx is None and isinstance(state, dict):
        ctx = state.get("context", [])
    return {
        "context": list(ctx or []),
        "pending_mmr_refetch": False,
    }
