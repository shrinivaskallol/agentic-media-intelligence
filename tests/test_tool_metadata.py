"""Tests for LLM-facing tool metadata (MCP tools and structured-output schemas)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))


def test_extraction_schema_intent_is_literal_enum() -> None:
    from app.nodes.extraction import ExtractionSchema

    schema = ExtractionSchema.model_json_schema()
    intent = schema["properties"]["intent"]
    assert intent["enum"] == ["RESEARCH", "COMPETITION", "IRRELEVANT"]


def test_grader_output_documents_provider_supplier_rule() -> None:
    from app.nodes.grader import GraderOutput

    desc = GraderOutput.model_json_schema()["properties"]["sufficient"]["description"]
    assert "provider" in desc.lower() or "supplier" in desc.lower()


def test_synthesis_status_and_critique_fields_documented() -> None:
    from app.nodes.synthesis import SynthesisStructuredOutput

    props = SynthesisStructuredOutput.model_json_schema()["properties"]
    assert "insufficient_data" in props["status"]["description"]
    assert "insufficient_data" in props["requires_critique"]["description"]


@pytest.mark.asyncio
async def test_mcp_tools_have_parameter_descriptions() -> None:
    from src import mcp_server

    tools = await mcp_server.mcp.list_tools()
    by_name = {t.name: t for t in tools}

    assert "run_market_research" in by_name
    assert "research_company" not in by_name

    run_tool = by_name["run_market_research"]
    run_params = run_tool.parameters
    assert run_params["properties"]["query"]["description"]
    assert run_params["properties"]["thread_id"]["description"]
    desc = run_tool.description or ""
    assert "WHEN TO USE" in desc
    assert "RETURNS" in desc.upper()

    resume_tool = by_name["resume_research"]
    resume_params = resume_tool.parameters
    assert resume_params["properties"]["thread_id"]["description"]
    lam = resume_params["properties"]["lambda_value"]
    assert lam["description"]
    assert lam.get("minimum") == 0.0
    assert lam.get("maximum") == 1.0
