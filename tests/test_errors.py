"""Tests for structured ToolError framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.errors.helpers import (
    classify_exception,
    empty_result,
    finalize_mcp_workflow_response,
    format_mcp_error,
    merge_retrieval_errors,
    rate_limit_error,
    service_unavailable_error,
    validation_error,
)
from app.errors.models import RetrievalSlice, ToolError
from app.nodes.extraction import ExtractionSchema
from app.nodes.grader import GraderOutput


def test_tool_error_schema() -> None:
    err = validation_error("bad input", field="query")
    assert err.is_error is True
    assert err.error_category == "validation"
    assert err.is_retryable is False
    dumped = err.model_dump()
    assert dumped["message"] == "bad input"


def test_classify_rate_limit() -> None:
    err = classify_exception(RuntimeError("429 Too Many Requests TPM exceeded"))
    assert err.error_category == "rate_limit"
    assert err.is_retryable is True


def test_classify_connection_as_service_unavailable() -> None:
    err = classify_exception(ConnectionError("Connection refused"))
    assert err.error_category == "service_unavailable"


def test_finalize_mcp_never_empty_on_failure() -> None:
    raw = finalize_mcp_workflow_response(
        {
            "response": "",
            "last_error": service_unavailable_error("db down").model_dump(),
        },
        "t1",
    )
    parsed = json.loads(raw)
    assert parsed["status"] == "error"
    assert parsed["error"]["error_category"] == "service_unavailable"


def test_finalize_mcp_no_evidence_is_empty_result_not_error() -> None:
    raw = finalize_mcp_workflow_response(
        {
            "response": "",
            "synthesis_status": "no_evidence",
        },
        "t1",
    )
    parsed = json.loads(raw)
    assert parsed["status"] == "empty_result"
    assert parsed["empty_result"]["result_kind"] == "no_evidence"
    assert parsed.get("error") is None


def test_finalize_mcp_success_markdown() -> None:
    out = finalize_mcp_workflow_response({"response": "## Report\n\nHello"}, "t1")
    assert out.startswith("## Report")


def test_finalize_mcp_interrupt() -> None:
    raw = finalize_mcp_workflow_response({"__interrupt__": ["pause"]}, "t1")
    parsed = json.loads(raw)
    assert parsed["status"] == "interrupted"


def test_merge_retrieval_errors_only_failures() -> None:
    ok = RetrievalSlice(items=["fact"], empty=False)
    bad = RetrievalSlice(
        items=[],
        error=service_unavailable_error("neo4j down", backend="neo4j"),
    )
    errs = merge_retrieval_errors(ok, bad)
    assert len(errs) == 1
    assert errs[0]["error_category"] == "service_unavailable"


def test_extraction_intent_enum_unchanged() -> None:
    schema = ExtractionSchema.model_json_schema()
    assert "RESEARCH" in schema["properties"]["intent"]["enum"]


def test_grader_provider_rule_in_schema() -> None:
    desc = GraderOutput.model_json_schema()["properties"]["sufficient"]["description"]
    assert "supplier" in desc.lower()


def test_empty_query_validation_payload() -> None:
    """MCP tools reject blank query before workflow (same JSON shape as run_market_research)."""
    raw = format_mcp_error(
        validation_error("query must be a non-empty string", field="query"),
        thread_id="t",
    )
    parsed = json.loads(raw)
    assert parsed["status"] == "error"
    assert parsed["error"]["error_category"] == "validation"


def test_format_mcp_error_json() -> None:
    raw = format_mcp_error(rate_limit_error("quota"), thread_id="x")
    parsed = json.loads(raw)
    assert parsed["status"] == "error"
    assert parsed["thread_id"] == "x"
