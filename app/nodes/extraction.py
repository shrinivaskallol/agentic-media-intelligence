"""
Entity extraction node: acts as the Gatekeeper.
Uses the LLM to decide if the query is worth the database "spend"
and extracts entities when intent is RESEARCH or COMPETITION.
Performs coreference resolution (e.g., "their" -> "Nvidia") when prior entities exist.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm_factory import get_llm

logger = logging.getLogger(__name__)
from app.prompts import get_prompt, get_system
from app.state.schema import GraphState

# Anaphora patterns that trigger query rewriting when we have prior context
ANAPHORA_PATTERNS = ("their", "its", "it", "they", "them", "his", "her", "this", "that", "those")


def _needs_rewrite(query: str, prior_entities: list[str]) -> bool:
    """True if query likely has anaphora and we have prior entities for resolution."""
    if not prior_entities:
        return False
    q = (query or "").lower()
    return any(p in q for p in ANAPHORA_PATTERNS)


def _rewrite_query(query: str, prior_entities: list[str], history: list[dict], llm) -> str:
    """Rephrase follow-up into a standalone search query using conversation context."""
    ctx = ", ".join(prior_entities) if prior_entities else "(none)"
    history_block = ""
    if history:
        history_str = ""
        for h in history[-3:]:
            q = h.get("query", "")
            r = (h.get("response", "") or "")[:200]
            history_str += f"\n  User: {q}\n  Assistant: {r}...\n"
        history_block = "Recent exchange:" + history_str

    prompt = get_prompt(
        "extraction", "coreference", ctx=ctx, history_block=history_block, query=query
    )
    result = llm.invoke(prompt)
    rewritten = (result.content if hasattr(result, "content") else str(result)).strip()
    return rewritten if rewritten else query


class ExtractionSchema(BaseModel):
    """Gatekeeper schema: entities, intent, and user-facing status."""

    entities: list[str] = Field(description="Companies or tech terms. Leave empty if IRRELEVANT.")
    intent: str = Field(description="Must be one of: RESEARCH, COMPETITION, or IRRELEVANT.")
    is_in_scope: bool = Field(
        default=True,
        description=(
            "False only if unrelated to semiconductor supply chains, finance, or market intelligence. "
            "True for semiconductor/company/supply-chain questions even if the named company is unknown, "
            "hypothetical, or non-existent (answer may be 'no data')."
        ),
    )
    log_message: str = Field(
        description="A brief status update for the user (e.g., 'Analyzing Nvidia supply chain...')."
    )


def _get_prior_context(state: GraphState | dict) -> tuple[list[str], list[dict]]:
    """Extract prior entities and history from state (from checkpoint in multi-turn)."""
    entities = getattr(state, "entities", None) or (
        state.get("entities", []) if isinstance(state, dict) else []
    )
    history = getattr(state, "history", None) or (
        state.get("history", []) if isinstance(state, dict) else []
    )
    return list(entities) if entities else [], list(history) if history else []


def entity_extractor(state: GraphState | dict) -> dict:
    """
    Gatekeeper node: classifies intent and extracts entities.
    Performs coreference resolution when prior entities exist and query has anaphora.
    """
    user_query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    prior_entities, history = _get_prior_context(state)

    # Coreference: rewrite "their" -> "Nvidia" when we have prior context
    query_for_extraction = user_query
    if _needs_rewrite(user_query, prior_entities):
        llm = get_llm()
        query_for_extraction = _rewrite_query(user_query, prior_entities, history, llm)
        if query_for_extraction != user_query:
            logger.info(
                "REWROTE QUERY (coreference): '%s...' -> '%s...'",
                user_query[:60],
                query_for_extraction[:60],
            )

    logger.info("ANALYZING QUERY: %s", query_for_extraction)

    llm = get_llm()
    structured_llm = llm.with_structured_output(ExtractionSchema)

    router_system = get_system("extraction", "router")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", router_system),
            ("human", "{messages}"),
        ]
    )

    result = (prompt | structured_llm).invoke({"messages": query_for_extraction})

    # Memory safeguard: when query has anaphora, ensure prior entities are included
    entities = list(result.entities) if result.entities else []
    if _needs_rewrite(user_query, prior_entities) and prior_entities:
        existing_lower = {e.lower() for e in entities}
        for pe in prior_entities:
            if pe and pe.strip() and pe.lower() not in existing_lower:
                entities.insert(0, pe)
                existing_lower.add(pe.lower())

    # Normalize intent (LLM may return "Irrelevant", "irrelevant", etc.)
    intent = str(result.intent or "").strip().upper()
    if intent not in ("RESEARCH", "COMPETITION", "IRRELEVANT"):
        intent = "IRRELEVANT" if not entities else "RESEARCH"

    logger.info("INTENT: %s | ENTITIES: %s | IN_SCOPE: %s", intent, entities, result.is_in_scope)
    logger.info("SYSTEM STATUS: %s", result.log_message)

    out: dict = {
        "entities": entities,
        "intent": intent,
    }

    if not getattr(result, "is_in_scope", True):
        out["exit_reason"] = "out_of_scope"
        out["entities"] = []
        out["intent"] = "RESEARCH"
        return out

    # If irrelevant, set response so user gets a polite message when we short-circuit
    if intent == "IRRELEVANT":
        out["response"] = (
            "This query is outside our market intelligence domain. "
            "I can help with industry, tech, or company research."
        )

    return out


# Legacy alias for backwards compatibility
extract_entities_node = entity_extractor
