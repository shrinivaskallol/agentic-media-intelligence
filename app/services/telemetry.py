"""
Telemetry for graph outcomes: exit_reason, partial answers, revisions.

Logs one structured JSON line per run (AMI_TELEMETRY) to the root logger — typically stderr in dev,
forwardable to files/ELK/Datadog in production for refusal-rate and KG-coverage dashboards.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def log_workflow_outcome(final_state: dict | None, *, hitl_interrupted: bool | None = None) -> None:
    """
    Emit a single parseable line for analytics: marketing fit (out_of_scope rate),
    knowledge gaps (max_retries_exceeded), and quality signals.

    ``hitl_interrupted``: when True/False, records whether the run stopped at diversity HITL
    (synthesis may not have run). Omit when unknown.
    """
    if not final_state:
        return
    er = (final_state.get("exit_reason") or "").strip() or None
    payload: dict = {
        "event": "workflow_outcome",
        "exit_reason": er,
        "partial_answer": bool(final_state.get("partial_answer")),
        "retrieval_revision_count": int(final_state.get("retrieval_revision_count") or 0),
        "revision_count": int(final_state.get("revision_count") or 0),
        "is_refused": bool(final_state.get("is_refused")),
        "critique_score": float(final_state.get("critique_score") or 0.0),
        "query_preview": (final_state.get("query") or "")[:200],
    }
    if hitl_interrupted is not None:
        payload["hitl_interrupted"] = bool(hitl_interrupted)
    logger.info("AMI_TELEMETRY %s", json.dumps(payload, default=str))


def log_graph_metrics(final_state: dict) -> None:
    """Backward-compatible alias; prefer ``log_workflow_outcome``."""
    log_workflow_outcome(final_state)
