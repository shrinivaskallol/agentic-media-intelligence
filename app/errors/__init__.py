"""Structured errors for MCP tools, graph nodes, and external integrations."""

from app.errors.exceptions import LLMInvocationError
from app.errors.helpers import (
    auth_error,
    classify_exception,
    empty_result,
    error_to_dict,
    error_to_state_patch,
    finalize_mcp_workflow_response,
    format_mcp_error,
    format_mcp_payload,
    internal_error,
    merge_retrieval_errors,
    not_found_error,
    rate_limit_error,
    retrieval_slice_from_exception,
    service_unavailable_error,
    timeout_error,
    tool_response_from_state,
    validation_error,
)
from app.errors.llm_invoke import llm_failure_patch
from app.errors.models import (
    ConnectionSlice,
    EmptyResult,
    ErrorCategory,
    RetrievalSlice,
    ToolError,
    ToolResponse,
)

__all__ = [
    "ConnectionSlice",
    "EmptyResult",
    "LLMInvocationError",
    "ErrorCategory",
    "RetrievalSlice",
    "ToolError",
    "ToolResponse",
    "auth_error",
    "classify_exception",
    "empty_result",
    "error_to_dict",
    "error_to_state_patch",
    "finalize_mcp_workflow_response",
    "format_mcp_error",
    "format_mcp_payload",
    "internal_error",
    "llm_failure_patch",
    "merge_retrieval_errors",
    "not_found_error",
    "rate_limit_error",
    "retrieval_slice_from_exception",
    "service_unavailable_error",
    "timeout_error",
    "tool_response_from_state",
    "validation_error",
]
