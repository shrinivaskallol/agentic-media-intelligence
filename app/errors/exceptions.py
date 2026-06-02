"""Typed exceptions carrying structured ToolError payloads."""

from __future__ import annotations

from app.errors.models import ToolError


class LLMInvocationError(Exception):
    """Raised when all LLM candidates fail or a non-retryable provider error occurs."""

    def __init__(self, tool_error: ToolError):
        self.tool_error = tool_error
        super().__init__(tool_error.message)
