"""
Guided refusal for out-of-scope queries: domain explanation + pivot suggestions.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from app.llm_factory import get_llm

logger = logging.getLogger(__name__)


REFUSAL_PROMPT = """You are a market-intelligence assistant. The user's query is outside your domain.

User query: {query}

Write a short, professional reply that:
1. Acknowledges the query is outside your scope.
2. States your domain clearly: semiconductor supply chains, finance-relevant market intelligence, and related tech/industry research.
3. Suggests exactly two concrete pivot questions the user could ask instead (related to semiconductors, foundries, equipment, or supply chain economics).

Use Markdown. Be concise (under 200 words)."""


def refusal_node(state: object) -> dict:
    """Produce guided refusal text; state must have exit_reason out_of_scope from extractor."""
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    logger.info("REFUSAL: guided pivot for out-of-scope query")

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", REFUSAL_PROMPT),
        ]
    )
    result = (prompt | llm).invoke({"query": query or "(empty)"})
    text = result.content if hasattr(result, "content") else str(result)
    return {
        "response": text.strip(),
        "exit_reason": "out_of_scope",
    }
