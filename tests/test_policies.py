"""Unit tests for deterministic policy hooks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.policies.escalation import detect_human_request, evaluate_escalation
from app.policies.evidence import count_evidence_units, meets_min_evidence
from app.policies.retrieval import apply_retrieval_budget, check_retrieval_circuit_breaker
from app.policies.synthesis_hooks import check_citation_coverage, check_closed_world_before_synthesis
from app.state.schema import GraphState


def test_apply_retrieval_budget_caps_entities() -> None:
    entities = [f"E{i}" for i in range(20)]
    capped, graph_lim, vec_lim = apply_retrieval_budget(entities, "RESEARCH")
    assert len(capped) <= 5
    assert graph_lim >= 1
    assert vec_lim >= 1


def test_retrieval_circuit_breaker_blocks_at_max() -> None:
    state = {"retrieval_revision_count": 3, "context": ["existing"]}
    patch = check_retrieval_circuit_breaker(state)
    assert patch is not None
    assert patch["context"] == ["existing"]


def test_retrieval_circuit_breaker_allows_below_max() -> None:
    assert check_retrieval_circuit_breaker({"retrieval_revision_count": 1}) is None


def test_meets_min_evidence_graph_or_vector() -> None:
    ctx_graph = ["--- GRAPH KNOWLEDGE ---\nGRAPH FACT: A PARTNERS_WITH B"]
    assert meets_min_evidence({"context": ctx_graph}) is True

    ctx_vec = [
        "--- TEXT CHUNKS ---\n"
        "[Vector-abc] [ORG] Nvidia: line one\n"
        "[Vector-def] [ORG] AMD: line two"
    ]
    assert meets_min_evidence({"context": ctx_vec}) is True

    ctx_weak = ["--- TEXT CHUNKS ---\n[Vector-abc] [ORG] Nvidia: only one"]
    assert meets_min_evidence({"context": ctx_weak}) is False


def test_count_evidence_units() -> None:
    ctx = [
        "--- GRAPH KNOWLEDGE ---\nGRAPH FACT: X\nGRAPH FACT: Y",
        "--- TEXT CHUNKS ---\n[Vector-a] [ORG] Z: text",
    ]
    g, v = count_evidence_units({"context": ctx})
    assert g == 2
    assert v == 1


def test_closed_world_blocks_empty_evidence() -> None:
    block = check_closed_world_before_synthesis(
        GraphState(query="q", context=[], partial_answer=False)
    )
    assert block is not None
    assert block["synthesis_status"] == "no_evidence"


def test_closed_world_allows_partial_mode() -> None:
    assert (
        check_closed_world_before_synthesis(
            GraphState(query="q", context=[], partial_answer=True)
        )
        is None
    )


def test_citation_coverage_requires_brackets() -> None:
    ok, _ = check_citation_coverage(
        "## Summary\n\n"
        "Nvidia uses TSMC 4NP for advanced packaging and HBM integration in datacenter GPUs.",
        min_ratio=0.5,
    )
    assert ok is False

    ok2, _ = check_citation_coverage(
        "## Summary\n\n"
        "Nvidia uses TSMC 4NP for advanced packaging [1] and HBM integration [2] in datacenter GPUs.",
        min_ratio=0.5,
    )
    assert ok2 is True


def test_detect_human_request() -> None:
    assert detect_human_request("Please speak to a human about this report") is True
    assert detect_human_request("What is TSMC 4NP capacity?") is False


def test_evaluate_escalation_user_request() -> None:
    esc, reason = evaluate_escalation({"query": "I need to talk to a human agent"})
    assert esc is True
    assert reason == "user_requested_human"


def test_evaluate_escalation_max_revisions() -> None:
    esc, reason = evaluate_escalation({"revision_count": 3, "query": "q"})
    assert esc is True
    assert reason == "max_refinement_cycles_exceeded"


def test_evaluate_escalation_infrastructure_failure_max_revisions() -> None:
    err = {
        "is_error": True,
        "error_category": "rate_limit",
        "is_retryable": True,
        "message": "429",
    }
    esc, reason = evaluate_escalation(
        {"revision_count": 3, "query": "q", "last_error": err}
    )
    assert esc is True
    assert reason == "infrastructure_failure_max_revisions"


def test_evaluate_escalation_infrastructure_failure_retrieval() -> None:
    err = {
        "is_error": True,
        "error_category": "service_unavailable",
        "is_retryable": True,
        "message": "neo4j down",
    }
    esc, reason = evaluate_escalation(
        {
            "retrieval_revision_count": 3,
            "exit_reason": "max_retries_exceeded",
            "last_error": err,
            "query": "q",
        }
    )
    assert esc is True
    assert reason == "infrastructure_failure_retrieval_exhausted"


@pytest.mark.asyncio
async def test_human_request_routes_without_retrieve() -> None:
    from app.logic.workflow import make_initial_state

    def fake_extractor(state):
        return {"entities": ["Nvidia"], "intent": "RESEARCH"}

    with patch("app.graph.entity_workflow.entity_extractor", fake_extractor):
        from app.graph.entity_workflow import build_workflow

        app = build_workflow()
        inputs = make_initial_state("Please speak to a human about Nvidia supply chain")
        nodes: list[str] = []
        async for chunk in app.astream(inputs):
            for name in chunk.keys():
                nodes.append(name)
        final = await app.ainvoke(inputs)
        fs = final.model_dump() if hasattr(final, "model_dump") else dict(final or {})

    assert "human_review" in nodes
    assert "retrieve" not in nodes
    assert fs.get("exit_reason") == "requires_human_review"
    assert fs.get("requires_human_review") is True
