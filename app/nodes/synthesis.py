"""
Synthesis node: produces final answer from query and retrieved context.
Uses intent-driven personas (RESEARCH vs COMPETITION) for dynamic report style.
Routed post-synthesis via structured fields (synthesis_status / requires_critique), not magic strings.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Literal

from app.errors.helpers import error_to_state_patch, internal_error
from app.errors.llm_invoke import llm_failure_patch
from app.llm_factory import get_llm
from app.policies.synthesis_hooks import (
    check_citation_coverage,
    check_closed_world_before_synthesis,
)
from app.prompts import get_prompt, get_system
from app.state.schema import GraphState
from app.utils.citations import build_numbered_context_for_llm
from app.utils.routing import normalize_llm_synthesis_status

logger = logging.getLogger(__name__)


class SynthesisStructuredOutput(BaseModel):
    """Machine-verifiable routing + user report — avoids parsing free-text REFUSAL tokens."""

    report_markdown: str = Field(
        ...,
        description="Full user-visible Markdown report (Executive Summary, sections, citations [1], [2]).",
    )
    status: Literal["complete", "insufficient_data"] = Field(
        ...,
        description=(
            "'complete': grounded report possible from RETRIEVED CONTEXT. "
            "'insufficient_data': context irrelevant or too weak to answer—set requires_critique "
            "true so the graph can self-correct. Not for out-of-scope queries (handled at extraction)."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Confidence 0–1 that RETRIEVED CONTEXT supports the query.",
    )
    requires_critique: bool = Field(
        default=False,
        description=(
            "True when status is insufficient_data or confidence < 0.7. "
            "False when status is complete and evidence clearly supports the answer."
        ),
    )


def synthesis_node(state: GraphState | dict) -> dict:
    """
    Final stage of the workflow. Synthesizes Graph and Vector data
    into a structured intelligence report. Persona varies by intent.
    If critique feedback exists and is not PASS, incorporates it as revision instructions.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    intent = getattr(state, "intent", None) or (
        state.get("intent", "RESEARCH") if isinstance(state, dict) else "RESEARCH"
    )
    intent = str(intent).strip().upper() or "RESEARCH"
    critique_feedback = getattr(state, "critique", None) or (
        state.get("critique", "") if isinstance(state, dict) else ""
    )
    partial_answer = getattr(state, "partial_answer", None) or (
        state.get("partial_answer", False) if isinstance(state, dict) else False
    )
    exit_reason = getattr(state, "exit_reason", None) or (
        state.get("exit_reason", "") if isinstance(state, dict) else ""
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

    numbered_evidence = build_numbered_context_for_llm(state)
    nc = numbered_evidence.strip()
    no_evidence = (not nc) or (nc == "No relevant data retrieved.")

    closed_world_block = check_closed_world_before_synthesis(state)
    if closed_world_block is not None:
        return closed_world_block

    if (partial_answer or exit_reason == "max_retries_exceeded") and no_evidence:
        logger.info(
            "SYNTHESIS: skipping LLM — no indexed evidence after max retrieval retries "
            "(exit_reason=%s)",
            exit_reason or "",
        )
        return {
            "response": (
                "## No evidence retrieved\n\n"
                "After multiple retrieval attempts, no relevant documents were found in the "
                "knowledge base for this query. A grounded report cannot be produced.\n\n"
                "(No indexed evidence matched your query.)"
            ),
            "is_refused": True,
            "synthesis_status": "no_evidence",
            "synthesis_confidence": 0.0,
            "synthesis_requires_critique": False,
        }

    partial_preamble = ""
    if partial_answer or exit_reason == "max_retries_exceeded":
        partial_preamble = (
            "PARTIAL ANSWER MODE: Retrieval did not yield fully sufficient context after maximum "
            "rewrite attempts. Produce the best possible structured report from the evidence below. "
            "State clearly at the start that the analysis is partial and evidence is sparse. "
            "Do not invent facts.\n\n"
        )
        logger.info("SYNTHESIZING PARTIAL REPORT (max_retries_exceeded)")

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

    context_str = numbered_evidence
    if partial_preamble:
        context_str = partial_preamble + context_str

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", selected_prompt),
            ("system", closed_world_instruction),
            ("system", f"Strict Rule: Use Markdown formatting. {entity_tag_rules}"),
            (
                "system",
                "You MUST fill the structured output schema: report_markdown, status, confidence, "
                "requires_critique. Never encode routing via special phrases inside report_markdown.",
            ),
            ("human", human_template),
        ]
    )

    structured_llm = llm.with_structured_output(SynthesisStructuredOutput)
    chain = prompt | structured_llm

    result: SynthesisStructuredOutput | None = None
    status: str | None = None
    for attempt in range(2):
        try:
            result = chain.invoke(
                {
                    "query": query,
                    "intent": intent,
                    "context": context_str,
                    "revision_instruction": revision_instruction,
                }
            )
        except (TypeError, ValueError, OSError, RuntimeError, ValidationError) as e:
            logger.warning(
                "SYNTHESIS structured invoke failed (attempt %s): %s",
                attempt + 1,
                e,
            )
            result = None
            llm_err_patch = llm_failure_patch(e, component="synthesis")
            if attempt == 1:
                stub = (
                    "## Report unavailable\n\n"
                    "The synthesis step could not complete due to an upstream language model error."
                )
                return {
                    "response": stub,
                    "synthesis_status": "invalid_structured_output",
                    "synthesis_confidence": 0.0,
                    "synthesis_requires_critique": True,
                    **llm_err_patch,
                }
            continue
        status = normalize_llm_synthesis_status(getattr(result, "status", None))
        if status is not None:
            break
        logger.warning(
            "SYNTHESIS invalid status field after attempt %s: %r",
            attempt + 1,
            getattr(result, "status", None),
        )

    if status is None or result is None:
        logger.error(
            "SYNTHESIS: invalid structured output after retries — forcing critique + safe stub"
        )
        stub = (
            "## Report unavailable\n\n"
            "The synthesis step did not return a valid structured verdict. "
            "The pipeline will attempt one self-correction cycle."
        )
        last_markdown = ""
        if result is not None:
            last_markdown = (getattr(result, "report_markdown", None) or "").strip()
        patch = {
            "response": last_markdown or stub,
            "synthesis_status": "invalid_structured_output",
            "synthesis_confidence": 0.0,
            "synthesis_requires_critique": True,
        }
        patch.update(
            error_to_state_patch(
                internal_error(
                    "Synthesis structured output invalid after retries.",
                    node="synthesis",
                )
            )
        )
        return patch

    needs_critique = bool(result.requires_critique) or status == "insufficient_data"
    try:
        confidence = float(result.confidence) if result.confidence is not None else 1.0
    except (TypeError, ValueError):
        confidence = 0.0
    answer = (result.report_markdown or "").strip()

    if status == "complete" and answer:
        cites_ok, cite_reason = check_citation_coverage(answer)
        if not cites_ok:
            logger.warning("POLICY PostSynthesis: citation coverage failed — %s", cite_reason)
            needs_critique = True

    logger.info(
        "SYNTHESIS structured: status=%s confidence=%.2f requires_critique=%s",
        status,
        confidence,
        needs_critique,
    )

    return {
        "response": answer,
        "synthesis_status": status,
        "synthesis_confidence": max(0.0, min(1.0, confidence)),
        "synthesis_requires_critique": needs_critique,
    }
