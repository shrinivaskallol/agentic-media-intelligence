"""
Grader node: Pre-synthesis binary check on whether retrieved context is sufficient.
Uses a fast LLM (temperature=0) for deterministic yes/no. Drives the agentic retrieval loop.
"""

import logging

from pydantic import BaseModel, Field

from app.llm_factory import get_critique_llm

logger = logging.getLogger(__name__)
from app.prompts import get_prompt
from app.state.schema import GraphState


class GraderOutput(BaseModel):
    """Binary verdict: is the retrieved context sufficient to answer the query?"""

    sufficient: bool = Field(
        description="True if the context contains enough information to answer the query; False otherwise.",
    )


def _get_context_str(state: GraphState | dict) -> str:
    """Extract context as a single string for the Grader."""
    context = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    if not context:
        return ""
    return "\n\n".join(str(c) for c in context if c)[:4000]


def grader_node(state: GraphState | dict) -> dict:
    """
    Grades whether retrieved context is sufficient to answer the query.
    Returns context_sufficient (bool) and optionally retrieval_revision_count (+1 when routing to rewrite).
    Uses fast model with temperature=0 for deterministic verdicts.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    context_str = _get_context_str(state)

    retrieval_rev = getattr(state, "retrieval_revision_count", None)
    if retrieval_rev is None and isinstance(state, dict):
        retrieval_rev = state.get("retrieval_revision_count", 0)
    retrieval_rev = int(retrieval_rev) if retrieval_rev is not None else 0

    if not context_str.strip():
        logger.info("GRADER: No context; insufficient")
        return {
            "context_sufficient": False,
            "retrieval_revision_count": 1 if retrieval_rev < 2 else 0,
        }

    llm = get_critique_llm()
    structured_llm = llm.with_structured_output(GraderOutput)

    prompt = get_prompt("grader", "main", query=query, context_str=context_str)
    result = structured_llm.invoke(prompt)
    sufficient = bool(result.sufficient) if hasattr(result, "sufficient") else False

    logger.info(
        "GRADER: context_sufficient=%s (retrieval_revision_count=%s)",
        sufficient,
        retrieval_rev,
    )

    out = {"context_sufficient": sufficient}
    if not sufficient and retrieval_rev < 2:
        out["retrieval_revision_count"] = 1

    return out
