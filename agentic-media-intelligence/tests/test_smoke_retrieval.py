"""
Smoke test for graph (Neo4j) and vector (Postgres) retrieval.
Requires: Docker (agent_graph, agent_sql) running and seeded.
Run: pytest tests/test_smoke_retrieval.py -v
"""

import pytest

from app.tools.graph_retriever import get_graph_context


@pytest.mark.integration
def test_graph_retrieval():
    """Graph (Neo4j) returns results or empty list."""
    entities = ["Nvidia", "TSMC", "GPU"]
    result = get_graph_context(entities, limit=15)
    assert isinstance(result, list)


@pytest.mark.integration
def test_vector_retrieval():
    """Vector (Postgres) returns results or empty list."""
    pytest.importorskip("sentence_transformers", reason="Vector needs sentence_transformers")
    from app.tools.retriever import get_vector_context

    result, _, _ = get_vector_context("Who are Nvidia's supply chain partners?", limit=5)
    assert isinstance(result, list)


@pytest.mark.integration
def test_at_least_one_source_operational():
    """At least graph or vector returns data (data layer operational)."""
    entities = ["Nvidia", "TSMC"]
    graph_bits = get_graph_context(entities, limit=5)
    vector_bits = []
    try:
        from app.tools.retriever import get_vector_context

        vector_bits, _, _ = get_vector_context("Nvidia TSMC", limit=3)
    except Exception:
        pass
    assert graph_bits or vector_bits, "Both sources empty; check Docker and ingestion"
