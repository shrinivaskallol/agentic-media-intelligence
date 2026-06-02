"""
Grader node: Pre-synthesis binary check on whether retrieved context is sufficient.
Uses a fast LLM (temperature=0) for deterministic yes/no. Drives the agentic retrieval loop.
"""

import logging

from pydantic import BaseModel, Field

from app.errors.llm_invoke import llm_failure_patch
from app.llm_factory import get_critique_llm
from app.policies.config import MAX_RETRIEVAL_REVISIONS
from app.policies.evidence import meets_min_evidence
from app.prompts import get_prompt
from app.state.schema import GraphState

logger = logging.getLogger(__name__)


class GraderOutput(BaseModel):
    """Binary verdict: is the retrieved context sufficient to answer the query?"""

    sufficient: bool = Field(
        description=(
            "True only if RETRIEVED CONTEXT explicitly supports answering the USER QUERY. "
            "For provider/supplier questions ('who supplies X', 'who provides Y'), the named "
            "provider/supplier entity must appear in context—not only downstream chain nodes "
            "(e.g. TSMC/ASML without the optics supplier when optics was asked). "
            "False triggers query rewrite and re-retrieval until policy max retries."
        ),
    )


def _get_context_str(state: GraphState | dict) -> str:
    """Extract context as a single string for the Grader."""
    context = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    if not context:
        return ""
    return "\n\n".join(str(c) for c in context if c)[:4000]


def _max_retrieval_partial_exit() -> dict:
    """
    Control-plane exit: router requires context_sufficient True to reach synthesis.
    Data-plane: partial_answer + exit_reason tell synthesis to do sparse / no-evidence report.
    """
    return {
        "context_sufficient": True,
        "partial_answer": True,
        "exit_reason": "max_retries_exceeded",
    }


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

    # Empty context: no LLM grade. After max retries, force synthesis (partial / no-results) — loop killer.
    if not context_str.strip():
        if retrieval_rev >= MAX_RETRIEVAL_REVISIONS:
            logger.warning(
                "GRADER: Context still empty after max retrieval retries (%s). Forcing partial exit.",
                retrieval_rev,
            )
            logger.info("GRADER: max retrieval retries — partial answer path (empty context)")
            return _max_retrieval_partial_exit()
        logger.info("GRADER: No context; insufficient")
        return {
            "context_sufficient": False,
            "retrieval_revision_count": 1,
        }

    if not meets_min_evidence(state):
        logger.info(
            "POLICY PostRetrieve: min evidence not met (graph/vector thresholds); insufficient"
        )
        if retrieval_rev >= MAX_RETRIEVAL_REVISIONS:
            logger.info("GRADER: max retrieval retries after min-evidence failure — partial path")
            return _max_retrieval_partial_exit()
        return {
            "context_sufficient": False,
            "retrieval_revision_count": 1,
        }

    llm = get_critique_llm()
    structured_llm = llm.with_structured_output(GraderOutput)

    prompt = get_prompt("grader", "main", query=query, context_str=context_str)
    try:
        result = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("GRADER LLM failed: %s", e)
        patch = llm_failure_patch(e, component="grader")
        patch.update({"context_sufficient": False, "retrieval_revision_count": 1})
        return patch

    sufficient = bool(result.sufficient) if hasattr(result, "sufficient") else False

    logger.info(
        "GRADER: context_sufficient=%s (retrieval_revision_count=%s)",
        sufficient,
        retrieval_rev,
    )

    if not sufficient and retrieval_rev >= MAX_RETRIEVAL_REVISIONS:
        logger.info("GRADER: max retrieval retries — partial answer path")
        return _max_retrieval_partial_exit()

    out = {"context_sufficient": sufficient}
    if not sufficient and retrieval_rev < MAX_RETRIEVAL_REVISIONS:
        out["retrieval_revision_count"] = 1

    return out
