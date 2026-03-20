"""
Rewrite node: When the Grader says context is insufficient, rewrite the query to be more specific.
Produces a sharper query and suggested entities for the next retrieval pass.
"""

import logging

from pydantic import BaseModel, Field

from app.llm_factory import get_llm

logger = logging.getLogger(__name__)
from app.prompts import get_prompt
from app.state.schema import GraphState


class RewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(
        description="A more specific, search-friendly version of the original query."
    )
    suggested_entities: list[str] = Field(
        default_factory=list,
        description="Key entities to search for (companies, products, technologies).",
    )


def rewrite_node(state: GraphState | dict) -> dict:
    """
    Rewrites the user query to be more specific for retrieval.
    E.g., "Apple optics" -> "Identify the primary optics provider for Apple's 3nm chip manufacturer (TSMC)."
    Also suggests entities to expand graph/vector search.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    entities = getattr(state, "entities", None) or (
        state.get("entities", []) if isinstance(state, dict) else []
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(RewriteOutput)

    entities_str = ", ".join(str(e) for e in entities) if entities else "(none)"
    prompt = get_prompt("rewrite", "main", query=query, entities=entities_str)
    result = structured_llm.invoke(prompt)
    rewritten = result.rewritten_query.strip() if hasattr(result, "rewritten_query") else query
    suggested = result.suggested_entities if hasattr(result, "suggested_entities") else entities

    if not rewritten:
        rewritten = query

    logger.info("REWRITE: '%s...' -> '%s...'", query[:50], rewritten[:80])

    return {
        "query": rewritten,
        "entities": suggested if suggested else entities,
    }
