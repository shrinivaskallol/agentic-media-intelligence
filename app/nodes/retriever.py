"""
Hybrid retrieval node: combines graph expansion (Neo4j) and vector search (Postgres).
Runs both in parallel. Behavior varies by intent (RESEARCH vs COMPETITION).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.state.schema import GraphState

logger = logging.getLogger(__name__)
from app.tools.retriever import get_graph_context, get_vector_context


def hybrid_retrieval_node(state: GraphState) -> dict:
    """
    Retrieve context from Neo4j (graph) and Postgres (pgvector) in parallel.
    Uses entities and intent from state.

    Intent-based behavior:
    - RESEARCH: Balanced—both Neo4j (relationships) and pgvector (news).
    - COMPETITION: Prioritize Neo4j (shared board members, supply chain overlaps);
      less pgvector.
    - IRRELEVANT: This node does not run (router short-circuits).
    """
    query = state.query
    entities = state.entities
    intent = getattr(state, "intent", None) or (
        state.get("intent", "") if isinstance(state, dict) else ""
    )
    intent = str(intent).strip().upper() or "RESEARCH"

    # Context window caps: Top 5 graph facts, Top 3 vector chunks (stay under Groq 6k TPM)
    if intent == "COMPETITION":
        graph_limit = 5
        vector_limit = 3
    else:
        # RESEARCH or fallback
        graph_limit = 5
        vector_limit = 3

    def _fetch_graph():
        return get_graph_context(entities, limit=graph_limit)

    def _fetch_vector():
        return get_vector_context(query, limit=vector_limit)

    # Run both retrievals in parallel
    graph_bits: list[str] = []
    vector_bits: list[str] = []
    chunk_ids: list[str] = []
    vector_metadata: list[dict] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_fetch_graph): "graph",
            ex.submit(_fetch_vector): "vector",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
                if name == "graph":
                    graph_bits = result
                else:
                    vector_bits, chunk_ids, vector_metadata = result
            except (OSError, ConnectionError, RuntimeError, TypeError, ValueError):
                pass  # Tools log their own warnings

    # Structured context with source IDs for closed-world grounding citations
    structured_parts: list[str] = []
    for i, bit in enumerate(vector_bits):
        cid = chunk_ids[i] if i < len(chunk_ids) else str(i)
        meta = vector_metadata[i] if i < len(vector_metadata) else {}
        etype = meta.get("entity_type", "TEXT")
        entity = meta.get("entity")
        prefix = f"[Vector-{cid}] "
        if entity:
            structured_parts.append(f"{prefix}[{etype}] {entity}: {bit}")
        else:
            structured_parts.append(f"{prefix}[{etype}] {bit}")

    graph_str = (
        "\n".join(f"GRAPH FACT: {f}" for f in graph_bits) if graph_bits else "(No graph results)"
    )
    vector_structured = structured_parts
    vector_str = "\n".join(vector_structured) if vector_structured else "(No vector results)"

    combined_context = [
        f"--- GRAPH KNOWLEDGE ---\n{graph_str}",
        f"--- TEXT CHUNKS (with entity types) ---\n{vector_str}",
    ]

    logger.debug(
        "RETRIEVED: vector_chunk_ids=%s, graph_facts=%d",
        chunk_ids,
        len(graph_bits),
    )
    for i, bit in enumerate(vector_bits[:5], 1):
        preview = (bit[:100] + "...") if len(bit) > 100 else bit
        cid = chunk_ids[i - 1] if i <= len(chunk_ids) else "?"
        logger.debug("  [%d] ID: %s | %s", i, cid, preview)

    retrieved_meta = [
        {"entity": m.get("entity"), "type": m.get("entity_type")} for m in vector_metadata
    ]
    return {
        "context": combined_context,
        "retrieved_ids": chunk_ids,
        "retrieved_metadata": retrieved_meta,
    }
