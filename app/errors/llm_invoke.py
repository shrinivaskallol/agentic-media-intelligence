"""Helpers for graph nodes calling LLM runnables with structured error patches."""

from __future__ import annotations

from typing import Any

from app.errors.exceptions import LLMInvocationError
from app.errors.helpers import classify_exception, error_to_state_patch


def llm_failure_patch(exc: BaseException, *, component: str = "llm") -> dict[str, Any]:
    """Return a graph state patch with ``last_error`` from an LLM or generic failure."""
    if isinstance(exc, LLMInvocationError):
        return error_to_state_patch(exc.tool_error)
    return error_to_state_patch(
        classify_exception(exc, component=component, operation="invoke")
    )
