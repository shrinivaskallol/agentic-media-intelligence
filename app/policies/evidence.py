"""Post-retrieval evidence rules (deterministic)."""

from __future__ import annotations

import re
from typing import Any

from app.policies.config import MIN_GRAPH_FACTS, MIN_VECTOR_CHUNKS

_GRAPH_FACT = re.compile(r"GRAPH FACT:\s*.+", re.IGNORECASE)
_VECTOR_LINE = re.compile(r"^\[Vector-", re.IGNORECASE | re.MULTILINE)
_NO_RESULTS = re.compile(r"\(No (?:graph|vector) results\)", re.IGNORECASE)


def _get_context(state: Any) -> list[str]:
    ctx = getattr(state, "context", None)
    if ctx is None and isinstance(state, dict):
        ctx = state.get("context", [])
    return list(ctx or [])


def count_evidence_units(state: Any) -> tuple[int, int]:
    """
    Return (graph_fact_count, vector_chunk_count) from structured context blocks.
    """
    graph_facts = 0
    vector_chunks = 0
    for block in _get_context(state):
        text = str(block)
        if _NO_RESULTS.search(text):
            continue
        graph_facts += len(_GRAPH_FACT.findall(text))
        vector_chunks += len(_VECTOR_LINE.findall(text))
    return graph_facts, vector_chunks


def meets_min_evidence(state: Any) -> bool:
    """
    Hard rule: at least MIN_GRAPH_FACTS graph facts OR MIN_VECTOR_CHUNKS vector chunks.
    Default: >=1 graph fact OR >=2 vector chunks.
    """
    graph_n, vec_n = count_evidence_units(state)
    if graph_n >= MIN_GRAPH_FACTS:
        return True
    if vec_n >= MIN_VECTOR_CHUNKS:
        return True
    return False
