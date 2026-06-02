"""
Hybrid retrieval node: combines graph expansion (Neo4j) and vector search (Postgres).
Runs both in parallel. Behavior varies by intent (RESEARCH vs COMPETITION).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.errors.helpers import error_to_dict, merge_retrieval_errors, service_unavailable_error
from app.errors.models import RetrievalSlice
from app.policies.retrieval import apply_retrieval_budget, check_retrieval_circuit_breaker
from app.state.schema import GraphState
from app.tools.retriever import get_graph_context, get_vector_context

logger = logging.getLogger(__name__)


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
    breaker_patch = check_retrieval_circuit_breaker(state)
    if breaker_patch is not None:
        return breaker_patch

    query = state.query
    entities = state.entities
    intent = getattr(state, "intent", None) or (
        state.get("intent", "") if isinstance(state, dict) else ""
    )
    intent = str(intent).strip().upper() or "RESEARCH"

    entities, graph_limit, vector_limit = apply_retrieval_budget(
        list(entities or []), intent
    )

    if isinstance(state, dict):
        raw_mmr = state.get("mmr_lambda", 1.0)
    else:
        raw_mmr = getattr(state, "mmr_lambda", 1.0)
    try:
        mmr_lambda = float(raw_mmr)
    except (TypeError, ValueError):
        mmr_lambda = 1.0
    mmr_lambda = max(0.0, min(1.0, mmr_lambda))

    def _fetch_graph() -> RetrievalSlice:
        return get_graph_context(entities, limit=graph_limit)

    def _fetch_vector() -> tuple[list[str], list[str], list[dict], RetrievalSlice | None]:
        bits, cids, meta, err_slice = get_vector_context(
            query, limit=vector_limit, mmr_lambda=mmr_lambda
        )
        return bits, cids, meta, err_slice

    graph_slice = RetrievalSlice(items=[])
    vector_bits: list[str] = []
    chunk_ids: list[str] = []
    vector_metadata: list[dict] = []
    vector_slice: RetrievalSlice | None = None

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
                    graph_slice = result
                else:
                    vector_bits, chunk_ids, vector_metadata, vector_slice = result
            except (OSError, ConnectionError, RuntimeError, TypeError, ValueError) as e:
                logger.warning("RETRIEVE %s task failed: %s", name, e)
                from app.errors.helpers import retrieval_slice_from_exception

                failed = retrieval_slice_from_exception(
                    e, backend=name, operation="hybrid_retrieval_node"
                )
                if name == "graph":
                    graph_slice = failed
                else:
                    vector_slice = failed

    graph_bits = graph_slice.items
    vector_result_slice = (
        vector_slice
        if vector_slice is not None
        else RetrievalSlice(items=vector_bits, empty=not vector_bits)
    )
    retrieval_errors = merge_retrieval_errors(graph_slice, vector_result_slice)

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
        "RETRIEVED: mmr_lambda=%.3f vector_chunk_ids=%s, graph_facts=%d",
        mmr_lambda,
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

    out: dict = {
        "context": combined_context,
        "retrieved_ids": chunk_ids,
        "retrieved_metadata": retrieved_meta,
        "pending_mmr_refetch": False,
        "retrieval_errors": retrieval_errors,
    }

    graph_failed = graph_slice.failed
    vector_failed = vector_slice is not None and vector_slice.failed
    if graph_failed and vector_failed:
        out["last_error"] = error_to_dict(
            service_unavailable_error(
                "Both graph and vector retrieval backends failed.",
                retrieval_errors=retrieval_errors,
            )
        )
    elif retrieval_errors:
        out["last_error"] = retrieval_errors[-1]

    return out
