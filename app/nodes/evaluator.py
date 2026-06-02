"""
Evaluator node: Quality Assurance Judge using Pydantic RagasMetrics.
Scores faithfulness, answer_relevancy, and context_precision. Collectible in LangSmith.
"""

import logging

from pydantic import BaseModel, Field, ValidationError

from app.errors.helpers import classify_exception, error_to_state_patch
from app.llm_factory import get_critique_llm
from app.prompts import get_prompt, get_system
from app.state.schema import GraphState
from app.utils.routing import synthesis_skip_faithfulness_judge

logger = logging.getLogger(__name__)


class RagasMetrics(BaseModel):
    """Structured metrics for LangSmith—collectible and filterable."""

    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="Score 0.0-1.0: Is the answer derived ONLY from the context?",
    )
    answer_relevancy: float = Field(
        ge=0.0,
        le=1.0,
        description="Score 0.0-1.0: Does the answer actually address the user's query?",
    )
    context_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Score 0.0-1.0: Was the retrieved context actually useful for the answer?",
    )
    critique_instruction: str = Field(
        default="",
        description="If any score is < 0.8, provide specific instructions for the Critique node.",
    )
    is_passing: bool = Field(
        description="True if all metrics meet the 0.8 threshold.",
    )


def _get_context_str(state: GraphState | dict) -> str:
    """Extract context as a single string for the Judge."""
    context = getattr(state, "context", None) or (
        state.get("context", []) if isinstance(state, dict) else []
    )
    if not context:
        return ""
    return "\n\n".join(str(c) for c in context if c)


async def evaluation_node(state: GraphState | dict) -> dict:
    """
    Judge node: Scores faithfulness, relevancy, precision via RagasMetrics.
    Returns eval_score (faithfulness) for loop logic, ragas_scores for LangSmith.
    """
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    response = getattr(state, "response", None) or (
        state.get("response", "") if isinstance(state, dict) else ""
    )
    context_str = _get_context_str(state)

    # Skip when no context or empty response (e.g. refusal)
    if not context_str.strip() or not (response or "").strip():
        logger.info("Skipping Evaluator: no context or empty response")
        return {
            "eval_score": 1.0,
            "ragas_scores": None,
            "is_refused": False,
        }

    if synthesis_skip_faithfulness_judge(state):
        return {
            "eval_score": 1.0,
            "ragas_scores": None,
            "is_refused": True,
        }

    try:
        llm = get_critique_llm()
        judge_chain = llm.with_structured_output(RagasMetrics)

        judge_system = get_system("evaluator", "judge_system")
        prompt = get_prompt(
            "evaluator",
            "judge_template",
            judge_system=judge_system,
            query=query,
            context_str=context_str,
            response=response,
        )
        metrics = await judge_chain.ainvoke(prompt)

        # Clip scores in case LLM exceeds bounds
        faithfulness = max(0.0, min(1.0, float(metrics.faithfulness)))
        answer_relevancy = max(0.0, min(1.0, float(metrics.answer_relevancy)))
        context_precision = max(0.0, min(1.0, float(metrics.context_precision)))

        # Derive pass/fail from scores so borderline runs (e.g. 0.8,0.8,0.8) are not
        # left failing when the Judge LLM sets is_passing inconsistently.
        is_passing = (
            faithfulness >= 0.8
            and answer_relevancy >= 0.8
            and context_precision >= 0.8
        )
        ragas_dict = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "critique_instruction": metrics.critique_instruction or "",
            "is_passing": is_passing,
        }

        # Refuse only after Critique has run at least once (revision_count >= 1).
        # Ensures is_passing=False always routes to Critique before we can give up.
        revision_count = getattr(state, "revision_count", None)
        if revision_count is None and isinstance(state, dict):
            revision_count = state.get("revision_count", 0)
        revision_count = int(revision_count) if revision_count is not None else 0

        is_refused = not is_passing and faithfulness >= 0.9 and revision_count >= 1

        logger.info(
            "RAGAS Judge: faithfulness=%.2f relevancy=%.2f precision=%.2f pass=%s",
            faithfulness,
            answer_relevancy,
            context_precision,
            is_passing,
        )

        return {
            "eval_score": faithfulness,
            "ragas_scores": ragas_dict,
            "critique_instruction": metrics.critique_instruction or "",
            "is_refused": is_refused,
        }
    except (ValueError, ValidationError, OSError, RuntimeError) as e:
        logger.warning("Evaluator Judge failed: %s", e)
        err = classify_exception(e, component="evaluator", operation="evaluation_node")
        return {
            "eval_score": 0.0,
            "ragas_scores": None,
            "critique_instruction": (
                "Evaluator unavailable; route to critique for manual quality check."
            ),
            "is_refused": False,
            **error_to_state_patch(err),
        }
