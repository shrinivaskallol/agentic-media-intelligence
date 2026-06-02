"""Structured error and outcome models for LLM-facing boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

ErrorCategory = Literal[
    "validation",
    "auth",
    "not_found",
    "timeout",
    "rate_limit",
    "service_unavailable",
    "internal_error",
]

OutcomeKind = Literal[
    "no_evidence",
    "no_results",
    "interrupted",
    "success",
]


class ToolError(BaseModel):
    """Infrastructure or operation failure — not a grounded 'no data' business outcome."""

    is_error: bool = True
    error_category: ErrorCategory
    is_retryable: bool
    message: str
    context: dict[str, Any] | None = None


class EmptyResult(BaseModel):
    """Successful call with no matching records (distinct from ToolError)."""

    is_error: bool = False
    result_kind: Literal["no_evidence", "no_results"] = "no_evidence"
    message: str
    context: dict[str, Any] | None = None


class ToolResponse(BaseModel):
    """Envelope for MCP / HTTP tool boundaries."""

    status: Literal["success", "error", "interrupted", "empty_result"] = "success"
    content: str | None = None
    error: ToolError | None = None
    empty_result: EmptyResult | None = None
    context: dict[str, Any] | None = None


class RetrievalSlice(BaseModel):
    """One retrieval backend result with optional structured failure."""

    items: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    empty: bool = Field(
        default=False,
        description="True when the backend responded successfully but returned no rows.",
    )

    @property
    def failed(self) -> bool:
        return self.error is not None


class StateErrorFields(TypedDict, total=False):
    """Graph state patch keys for structured errors."""

    last_error: dict[str, Any]
    retrieval_errors: list[dict[str, Any]]
