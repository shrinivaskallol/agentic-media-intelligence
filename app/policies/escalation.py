"""Deterministic human-escalation triggers (no sentiment / model-confidence)."""

from __future__ import annotations

import re
from typing import Any

from app.policies.config import (
    MAX_REFINEMENT_REVISIONS,
    MAX_RETRIEVAL_REVISIONS,
    MAX_SYNTHESIS_SELF_CORRECTION,
)
from app.policies.evidence import meets_min_evidence
from app.utils.routing import synthesis_norm_status

_HUMAN_REQUEST = re.compile(
    r"\b("
    r"speak to (?:a )?human|talk to (?:a )?(?:human|person|agent)|"
    r"human (?:agent|reviewer|analyst)|escalate(?:\s+to)?(?:\s+human)?|"
    r"real person|customer service"
    r")\b",
    re.IGNORECASE,
)


def detect_human_request(query: str) -> bool:
    """Valid escalation: user explicitly asks for a human."""
    return bool(_HUMAN_REQUEST.search(query or ""))


def _int_field(state: Any, name: str, default: int = 0) -> int:
    val = getattr(state, name, None)
    if val is None and isinstance(state, dict):
        val = state.get(name, default)
    return int(val) if val is not None else default


def evaluate_escalation(state: Any) -> tuple[bool, str]:
    """
    Return (should_escalate, reason_code). Only deterministic triggers.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    if detect_human_request(str(query)):
        return True, "user_requested_human"

    revision_count = _int_field(state, "revision_count")
    retrieval_rev = _int_field(state, "retrieval_revision_count")
    status = synthesis_norm_status(state)
    partial = bool(
        getattr(state, "partial_answer", None)
        or (state.get("partial_answer") if isinstance(state, dict) else False)
    )
    exit_reason = getattr(state, "exit_reason", None) or (
        state.get("exit_reason", "") if isinstance(state, dict) else ""
    )
    exit_reason = str(exit_reason or "").strip()

    # Business threshold: max refinement cycles without passing quality gate
    if revision_count >= MAX_REFINEMENT_REVISIONS:
        return True, "max_refinement_cycles_exceeded"

    # Retrieval exhausted with no usable evidence (not a successful partial report)
    if (
        retrieval_rev >= MAX_RETRIEVAL_REVISIONS
        and partial
        and exit_reason == "max_retries_exceeded"
        and status == "no_evidence"
    ):
        return True, "retrieval_exhausted_no_evidence"

    # Structured output repeatedly invalid
    if (
        status == "invalid_structured_output"
        and revision_count >= MAX_SYNTHESIS_SELF_CORRECTION
    ):
        return True, "structured_output_failure"

    # Policy gap: evidence exists but synthesis still insufficient after self-correction budget
    if (
        status == "insufficient_data"
        and revision_count >= MAX_SYNTHESIS_SELF_CORRECTION
        and meets_min_evidence(state)
    ):
        return True, "policy_gap_insufficient_grounding"

    return False, ""
