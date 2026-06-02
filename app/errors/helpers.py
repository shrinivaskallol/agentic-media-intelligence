"""Factories and serializers for structured agent/tool errors."""

from __future__ import annotations

import json
from typing import Any, Literal

from app.errors.models import EmptyResult, ErrorCategory, RetrievalSlice, ToolError, ToolResponse

_RETRYABLE: dict[ErrorCategory, bool] = {
    "validation": False,
    "auth": False,
    "not_found": False,
    "timeout": True,
    "rate_limit": True,
    "service_unavailable": True,
    "internal_error": False,
}


def _make(
    category: ErrorCategory,
    message: str,
    *,
    context: dict[str, Any] | None = None,
    is_retryable: bool | None = None,
) -> ToolError:
    return ToolError(
        error_category=category,
        is_retryable=is_retryable if is_retryable is not None else _RETRYABLE[category],
        message=message,
        context=context,
    )


def validation_error(message: str, **context: Any) -> ToolError:
    return _make("validation", message, context=context or None)


def auth_error(message: str, **context: Any) -> ToolError:
    return _make("auth", message, context=context or None)


def not_found_error(message: str, **context: Any) -> ToolError:
    return _make("not_found", message, context=context or None)


def timeout_error(message: str, **context: Any) -> ToolError:
    return _make("timeout", message, context=context or None)


def rate_limit_error(message: str, **context: Any) -> ToolError:
    return _make("rate_limit", message, context=context or None)


def service_unavailable_error(message: str, **context: Any) -> ToolError:
    return _make("service_unavailable", message, context=context or None)


def internal_error(message: str, **context: Any) -> ToolError:
    return _make("internal_error", message, context=context or None)


def empty_result(
    message: str,
    *,
    kind: Literal["no_evidence", "no_results"] = "no_evidence",
    **context: Any,
) -> EmptyResult:
    return EmptyResult(
        result_kind=kind,
        message=message,
        context=context or None,
    )


def classify_exception(
    exc: BaseException,
    *,
    component: str | None = None,
    operation: str | None = None,
) -> ToolError:
    """Map a raised exception to a ToolError category."""
    ctx: dict[str, Any] = {}
    if component:
        ctx["component"] = component
    if operation:
        ctx["operation"] = operation
    ctx["exception_type"] = type(exc).__name__

    msg = str(exc).strip() or type(exc).__name__
    upper = msg.upper()

    if isinstance(exc, TimeoutError):
        return timeout_error(msg, **ctx)
    if isinstance(exc, PermissionError):
        return auth_error(msg, **ctx)
    if isinstance(exc, (ConnectionError, OSError)):
        try:
            import psycopg2

            if isinstance(exc, psycopg2.OperationalError):
                return service_unavailable_error(
                    f"Database connection failed: {msg}", **ctx
                )
        except ImportError:
            pass
        try:
            from neo4j.exceptions import DriverError, ServiceUnavailable

            if isinstance(exc, (ServiceUnavailable, DriverError)):
                return service_unavailable_error(f"Graph database unavailable: {msg}", **ctx)
        except ImportError:
            pass
        return service_unavailable_error(msg, **ctx)
    if isinstance(exc, ValueError):
        return validation_error(msg, **ctx)

    if any(
        k in upper
        for k in (
            "429",
            "RATE_LIMIT",
            "RATELIMIT",
            "RESOURCE_EXHAUSTED",
            "QUOTA",
            "TPM",
        )
    ):
        return rate_limit_error(msg, **ctx)
    if any(k in upper for k in ("401", "403", "INVALID_API_KEY", "EXPIRED_API_KEY", "UNAUTHORIZED")):
        return auth_error(msg, **ctx)
    if any(k in upper for k in ("404", "NOT_FOUND", "NOT FOUND")):
        return not_found_error(msg, **ctx)
    if any(k in upper for k in ("TIMEOUT", "TIMED OUT", "DEADLINE")):
        return timeout_error(msg, **ctx)
    if any(k in upper for k in ("503", "UNAVAILABLE", "SERVICE_UNAVAILABLE", "CONNECTION REFUSED")):
        return service_unavailable_error(msg, **ctx)

    return internal_error(msg, **ctx)


def error_to_dict(error: ToolError) -> dict[str, Any]:
    return error.model_dump()


def error_to_state_patch(error: ToolError) -> dict[str, Any]:
    return {"last_error": error_to_dict(error)}


def merge_retrieval_errors(*slices: RetrievalSlice) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sl in slices:
        if sl.error is not None:
            d = error_to_dict(sl.error)
            d.setdefault("context", {})
            if isinstance(d["context"], dict):
                d["context"]["empty_after_call"] = sl.empty and not sl.items
            out.append(d)
    return out


def retrieval_slice_from_exception(
    exc: BaseException,
    *,
    backend: str,
    operation: str,
) -> RetrievalSlice:
    err = classify_exception(exc, component=backend, operation=operation)
    ctx = dict(err.context or {})
    ctx["backend"] = backend
    return RetrievalSlice(items=[], error=err.model_copy(update={"context": ctx}))


def format_mcp_payload(
    *,
    status: str,
    error: ToolError | None = None,
    empty_result_obj: EmptyResult | None = None,
    content: str | None = None,
    thread_id: str | None = None,
    interrupt: list | None = None,
    hint: str | None = None,
) -> str:
    """JSON string for MCP tool returns (errors, interrupts, empty outcomes)."""
    body: dict[str, Any] = {"status": status}
    if thread_id is not None:
        body["thread_id"] = thread_id
    if error is not None:
        body["error"] = error.model_dump()
    if empty_result_obj is not None:
        body["empty_result"] = empty_result_obj.model_dump()
    if content is not None:
        body["content"] = content
    if interrupt is not None:
        body["interrupt"] = interrupt
    if hint is not None:
        body["hint"] = hint
    return json.dumps(body, indent=2)


def format_mcp_error(error: ToolError, *, thread_id: str | None = None) -> str:
    return format_mcp_payload(status="error", error=error, thread_id=thread_id)


def finalize_mcp_workflow_response(state: dict[str, Any], thread_id: str) -> str:
    """
    Convert merged graph state to MCP return string.
    Never returns bare empty string on operational failure.
    """
    if state.get("__interrupt__"):
        intrs = state.get("__interrupt__") or []
        serialized = []
        for it in intrs:
            if hasattr(it, "value"):
                serialized.append({"value": it.value, "id": getattr(it, "id", None)})
            else:
                serialized.append(str(it))
        return format_mcp_payload(
            status="interrupted",
            thread_id=thread_id,
            interrupt=serialized,
            hint=(
                "Call resume_research(thread_id=..., lambda_value=0.0-1.0) "
                "with the same thread_id after run_market_research interrupted."
            ),
        )

    response = (state.get("response") or "").strip()
    last_error_raw = state.get("last_error")
    if last_error_raw and not response:
        try:
            err = ToolError.model_validate(last_error_raw)
        except Exception:
            err = internal_error(
                "Workflow failed with an unstructured error.",
                raw=last_error_raw,
            )
        return format_mcp_error(err, thread_id=thread_id)

    if not response:
        synthesis_status = str(state.get("synthesis_status") or "")
        if synthesis_status == "no_evidence":
            return format_mcp_payload(
                status="empty_result",
                thread_id=thread_id,
                empty_result_obj=empty_result(
                    "No grounded evidence in the knowledge base for this query.",
                    exit_reason=state.get("exit_reason"),
                    synthesis_status=synthesis_status,
                ),
                content=(state.get("response") or "").strip() or None,
            )
        return format_mcp_error(
            internal_error(
                "Workflow completed without a response.",
                thread_id=thread_id,
                exit_reason=state.get("exit_reason"),
                synthesis_status=synthesis_status,
            ),
            thread_id=thread_id,
        )

    if last_error_raw:
        # Non-fatal partial failure (e.g. one retrieval backend down) — success with response
        return response

    return response


def tool_response_from_state(state: dict[str, Any]) -> ToolResponse:
    """Build ToolResponse envelope from graph state (for HTTP / dashboard)."""
    if state.get("__interrupt__"):
        return ToolResponse(status="interrupted", context={"thread_id": state.get("thread_id")})

    response = (state.get("response") or "").strip()
    last_error_raw = state.get("last_error")

    if last_error_raw and not response:
        err = ToolError.model_validate(last_error_raw)
        return ToolResponse(status="error", error=err)

    if not response:
        if str(state.get("synthesis_status") or "") == "no_evidence":
            return ToolResponse(
                status="empty_result",
                empty_result=empty_result(
                    "No grounded evidence in the knowledge base for this query.",
                    synthesis_status="no_evidence",
                ),
            )
        return ToolResponse(
            status="error",
            error=internal_error(
                "Workflow completed without a response.",
                exit_reason=state.get("exit_reason"),
            ),
        )

    return ToolResponse(status="success", content=response)
