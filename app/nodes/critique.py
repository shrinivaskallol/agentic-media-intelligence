"""
Critique node: Red Team auditor that validates the synthesis report against context.
Uses temperature=0 and structured output for deterministic scoring (0.0-1.0).
"""

import logging

from pydantic import BaseModel, Field

from app.llm_factory import get_critique_llm

logger = logging.getLogger(__name__)
from app.prompts import get_prompt
from app.state.schema import GraphState


class CritiqueOutput(BaseModel):
    """Structured critique: score 0.0-1.0 and feedback text."""

    score: float = Field(
        description="Quality score 0.0 to 1.0. 1.0 = perfect, 0.0 = complete failure.",
        ge=0.0,
        le=1.0,
    )
    feedback: str = Field(
        description=(
            "Bulleted list of specific corrections, or 'PASS' if score >= 0.85 with no "
            "grounding or category errors."
        ),
    )


def _report(state: GraphState | dict) -> str:
    """Extract report (response) from state."""
    return getattr(state, "response", None) or (
        state.get("response", "") if isinstance(state, dict) else ""
    )


def _extract_top_chunks(ctx: list, top_n: int = 3) -> str:
    """
    Extract top N most relevant chunks for critique. Avoids sending full graph + all chunks
    (which can exceed Groq 8B TPM limits). Prioritizes text chunks over raw graph data.
    """
    if not ctx:
        return "No context."
    # ctx is typically [graph_section, text_chunks_section]
    text_section = None
    graph_section = None
    for item in ctx:
        s = str(item)
        if "--- TEXT CHUNKS" in s or "TEXT CHUNKS" in s:
            text_section = s
        elif "--- GRAPH KNOWLEDGE" in s or "GRAPH KNOWLEDGE" in s:
            graph_section = s
    parts = []
    # Top N chunks from the text section (lines like [TYPE] entity: content)
    if text_section:
        lines = [
            ln.strip()
            for ln in text_section.split("\n")
            if ln.strip() and not ln.strip().startswith("---")
        ]
        # Each chunk is typically one or more lines; approximate by taking first top_n*2 lines
        chunk_lines = lines[: top_n * 3]  # ~3 lines per chunk
        if chunk_lines:
            parts.append("--- TOP CHUNKS (for grounding audit) ---\n" + "\n".join(chunk_lines))
    # One-line graph summary if present (no raw triples to stay under token budget)
    if graph_section:
        glines = [ln for ln in graph_section.split("\n") if ln.strip() and "GRAPH FACT:" in ln][:5]
        if glines:
            parts.append("--- GRAPH SUMMARY (sample facts) ---\n" + "\n".join(glines[:3]))
    return "\n\n".join(parts) if parts else "No context."


def _critique_instruction(state: GraphState | dict) -> str:
    """Extract evaluator's critique_instruction for the Critique node."""
    return getattr(state, "critique_instruction", None) or (
        state.get("critique_instruction", "") if isinstance(state, dict) else ""
    )


def critique_node(state: GraphState | dict) -> dict:
    """
    Senior Technical Auditor: verifies the report against context.
    Returns structured score (0.0-1.0) and feedback. Threshold 0.85 for pass.
    Uses context budgeting: only report + top 3 chunks to stay under Groq TPM limits.
    If the Evaluator provided critique_instruction, it is included to focus the audit.
    """
    report = _report(state)
    ctx = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    context = _extract_top_chunks(ctx, top_n=3)
    eval_instruction = _critique_instruction(state)

    logger.info("CRITIQUE: Auditing report (score 0.0-1.0)")

    llm = get_critique_llm()
    structured_llm = llm.with_structured_output(CritiqueOutput)

    instruction_block = (
        f"\nEVALUATOR FOCUS (from Judge): {eval_instruction}\n" if eval_instruction else ""
    )
    prompt = get_prompt(
        "critique",
        "main",
        instruction_block=instruction_block,
        report=report,
        context=context,
    )
    result = structured_llm.invoke(prompt)
    score = float(result.score) if hasattr(result, "score") else 0.0
    feedback = result.feedback if hasattr(result, "feedback") else str(result)

    if score >= 0.85:
        logger.info("CRITIQUE: PASS (score=%.2f)", score)
    else:
        logger.info(
            "CRITIQUE: Score %.2f below threshold; refinement triggered",
            score,
        )

    preview = (feedback[:200] + "...") if len(feedback) > 200 else feedback
    logger.debug("CRITIQUE: score=%.2f | feedback=%s", score, preview)

    return {
        "critique": feedback,
        "critique_score": score,
        "revision_count": 1,
    }
