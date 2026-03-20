"""
Synthesis node: produces final answer from query and retrieved context.
Uses intent-driven personas (RESEARCH vs COMPETITION) for dynamic report style.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from app.llm_factory import get_llm

logger = logging.getLogger(__name__)
from app.prompts import get_prompt, get_system
from app.state.schema import GraphState
from app.utils.context_budget import budget_context_for_synthesis


def synthesis_node(state: GraphState | dict) -> dict:
    """
    Final stage of the workflow. Synthesizes Graph and Vector data
    into a structured intelligence report. Persona varies by intent.
    If critique feedback exists and is not PASS, incorporates it as revision instructions.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    context = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    intent = getattr(state, "intent", None) or (
        state.get("intent", "RESEARCH") if isinstance(state, dict) else "RESEARCH"
    )
    intent = str(intent).strip().upper() or "RESEARCH"
    critique_feedback = getattr(state, "critique", None) or (
        state.get("critique", "") if isinstance(state, dict) else ""
    )

    revision_instruction = ""
    if critique_feedback and critique_feedback.strip().upper() != "PASS":
        # Truncate long critique to stay under token limits
        fb = critique_feedback.strip()[:1200] + ("..." if len(critique_feedback) > 1200 else "")
        revision_instruction = (
            f"\n\nPREVIOUS CRITIQUE & REQUIRED FIXES:\n{fb}\n"
            "Please regenerate the report addressing these specific points."
        )
        logger.info("Regenerating with critique feedback")

    logger.info("SYNTHESIZING %s REPORT", intent)

    llm = get_llm()

    selected_prompt = (
        get_system("synthesis.personas", intent)
        if intent in ("RESEARCH", "COMPETITION")
        else get_system("synthesis.personas", "default")
    )
    closed_world_instruction = get_system("synthesis", "closed_world")
    entity_tag_rules = get_system("synthesis", "entity_rules")
    human_template = get_prompt(
        "synthesis",
        "human_template",
        query="{query}",
        intent="{intent}",
        context="{context}",
        revision_instruction="{revision_instruction}",
    )

    # Budget context to stay under Groq 8B TPM (~6k); Gemini has higher limits
    context_str = (
        budget_context_for_synthesis(context) if context else "No relevant data retrieved."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", selected_prompt),
            ("system", closed_world_instruction),
            ("system", f"Strict Rule: Use Markdown formatting. {entity_tag_rules}"),
            ("human", human_template),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "query": query,
            "intent": intent,
            "context": context_str,
            "revision_instruction": revision_instruction,
        }
    )
    answer = result.content if hasattr(result, "content") else str(result)

    return {"response": answer}
