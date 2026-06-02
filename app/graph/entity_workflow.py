"""
Entity extraction workflow: extract → [conditional] retrieve → synthesis → END.
Short-circuits to END when intent is IRRELEVANT (domain guardrail).
Persistent memory via SQLite checkpointer for stateful conversations.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from langgraph.graph import END, StateGraph

from app.nodes.critique import critique_node
from app.nodes.diversity_gate import diversity_gate_node
from app.nodes.evaluator import evaluation_node
from app.nodes.extraction import entity_extractor
from app.nodes.fallback import fallback_node
from app.nodes.grader import grader_node
from app.nodes.human_review import human_review_node
from app.nodes.refusal import refusal_node
from app.nodes.retriever import hybrid_retrieval_node
from app.nodes.rewrite import rewrite_node
from app.nodes.synthesis import synthesis_node
from app.state.schema import GraphState
from app.utils.node_timing import timed_node
from app.policies.escalation import detect_human_request, evaluate_escalation
from app.utils.routing import route_after_eval, synthesis_routes_to_critique


def _after_synthesis_route(state: GraphState | dict) -> str:
    """
    Route synthesis output via structured fields (insufficient_data, invalid_structured_output) or requires_critique → critique.
    """
    if synthesis_routes_to_critique(state):
        logger.info("SYNTHESIS: structured signal → Critique (self-correction)")
        return "critique"
    return "evaluator"


def _route_after_eval(state: GraphState | dict) -> str:
    """Delegate to routing module; add logging."""
    escalate, reason = evaluate_escalation(state)
    if escalate:
        logger.info("POLICY: escalating to human_review after eval (%s)", reason)
        return "human_review"

    result = route_after_eval(state)
    ragas = getattr(state, "ragas_scores", None) or (
        state.get("ragas_scores") if isinstance(state, dict) else None
    )
    score = getattr(state, "eval_score", None)
    if score is None and isinstance(state, dict):
        score = state.get("eval_score", 0.0)
    score = float(score) if score is not None else 0.0
    if result == "critique":
        reason = "is_passing=False" if ragas is not None else f"faithfulness={score:.2f}<0.8"
        logger.info("RAGAS: %s; routing to critique", reason)
    else:
        reason = "is_passing=True" if ragas is not None else f"faithfulness={score:.2f}>=0.8"
        logger.info("RAGAS: %s; finishing", reason)
    return result


def _should_continue(state: GraphState | dict) -> str:
    """
    Quality gate: synthesis self-correction branch or max revisions -> fallback; score >= 0.85 -> finish; else refine.
    """
    response = getattr(state, "response", None) or (
        state.get("response", "") if isinstance(state, dict) else ""
    )
    score = getattr(state, "critique_score", None)
    if score is None and isinstance(state, dict):
        score = state.get("critique_score", 0.0)
    score = float(score) if score is not None else 0.0

    count = getattr(state, "revision_count", None)
    if count is None and isinstance(state, dict):
        count = state.get("revision_count", 0)
    count = count if count is not None else 0

    escalate, reason = evaluate_escalation(state)
    if escalate:
        logger.info("POLICY: escalating to human_review from critique gate (%s)", reason)
        return "human_review"

    # Synthesis signals that need critique/refine before fallback (insufficient_data, invalid output, or explicit flag).
    if synthesis_routes_to_critique(state):
        if count >= 2:
            logger.info(
                "SYNTHESIS self-correction: max attempts for this branch; falling back"
            )
            return "fallback"
        logger.info("SYNTHESIS self-correction: forcing Critique → Synthesis refine")
        return "refine"
    if count >= 3:
        logger.info("HARD STOP: Max revisions (3) reached; falling back to evidence-only")
        return "fallback"

    # Pass only if score is high
    if score >= 0.85:
        return "finish"

    logger.info("CRITIQUE: Regenerating (revision %d/3)", count + 1)
    return "refine"


def _route_after_diversity(state: GraphState | dict) -> str:
    """After HITL sets λ, re-run retrieve for MMR; otherwise continue to grader."""
    pending = getattr(state, "pending_mmr_refetch", None)
    if pending is None and isinstance(state, dict):
        pending = state.get("pending_mmr_refetch", False)
    pending = bool(pending)
    if pending:
        logger.info("DIVERSITY: pending MMR re-fetch; routing to retrieve")
        return "re_retrieve"
    return "grader"


def _grader_route(state: GraphState | dict) -> str:
    """
    Agentic loop: rewrite until context sufficient or partial_answer after max retries.
    """
    partial = getattr(state, "partial_answer", None)
    if partial is None and isinstance(state, dict):
        partial = state.get("partial_answer", False)
    if bool(partial):
        logger.info("GRADER: Partial answer flag; proceeding to synthesis")
        return "generate_response"

    sufficient = getattr(state, "context_sufficient", None)
    if sufficient is None and isinstance(state, dict):
        sufficient = state.get("context_sufficient", True)
    sufficient = bool(sufficient) if sufficient is not None else True

    if sufficient:
        logger.info("GRADER: Context sufficient; proceeding to synthesis")
        return "generate_response"

    logger.info("GRADER: Context insufficient; routing to rewrite")
    return "rewrite_query"


def router_logic(state: GraphState | dict) -> str:
    """After extractor: out-of-scope → refusal; IRRELEVANT → END; else retrieve."""
    query = getattr(state, "query", None) or (
        state.get("query", "") if isinstance(state, dict) else ""
    )
    if detect_human_request(str(query or "")):
        logger.info("POLICY: user requested human — routing to human_review")
        return "human_review"

    exit_reason = getattr(state, "exit_reason", None) or (
        state.get("exit_reason", "") if isinstance(state, dict) else ""
    )
    if exit_reason == "out_of_scope":
        logger.info("ROUTING: Out of scope → guided refusal")
        return "refusal"

    intent = getattr(state, "intent", None) or (
        state.get("intent", "") if isinstance(state, dict) else ""
    )
    entities = getattr(state, "entities", None) or (
        state.get("entities", []) if isinstance(state, dict) else []
    )

    if intent == "IRRELEVANT":
        logger.info("SHORT-CIRCUIT: Query out of domain. Ending workflow.")
        return "end"

    logger.info("ROUTING: Proceeding to Retrieval for %s", entities)
    return "continue"


def build_workflow(checkpointer=None):
    """Build the extract → retrieve → synthesis entity workflow with conditional routing.

    For async persistence (ainvoke/astream), pass an AsyncSqliteSaver obtained from:
        async with AsyncSqliteSaver.from_conn_string(str(get_checkpoint_db_path())) as saver:
            app = build_workflow(checkpointer=saver)
            ...
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("extractor", timed_node(entity_extractor, "extractor"))
    workflow.add_node("refusal", timed_node(refusal_node, "refusal"))
    workflow.add_node("retrieve", timed_node(hybrid_retrieval_node, "retrieve"))
    workflow.add_node("diversity_gate", timed_node(diversity_gate_node, "diversity_gate"))
    workflow.add_node("grader", timed_node(grader_node, "grader"))
    workflow.add_node("rewrite", timed_node(rewrite_node, "rewrite"))
    workflow.add_node("synthesis", timed_node(synthesis_node, "synthesis"))
    workflow.add_node("evaluator", timed_node(evaluation_node, "evaluator"))
    workflow.add_node("critique", timed_node(critique_node, "critique"))
    workflow.add_node("fallback", timed_node(fallback_node, "fallback"))
    workflow.add_node("human_review", timed_node(human_review_node, "human_review"))

    workflow.set_entry_point("extractor")

    workflow.add_conditional_edges(
        "extractor",
        router_logic,
        {
            "continue": "retrieve",
            "end": END,
            "refusal": "refusal",
            "human_review": "human_review",
        },
    )
    workflow.add_edge("refusal", END)

    workflow.add_edge("retrieve", "diversity_gate")
    workflow.add_conditional_edges(
        "diversity_gate",
        _route_after_diversity,
        {
            "re_retrieve": "retrieve",
            "grader": "grader",
        },
    )
    workflow.add_conditional_edges(
        "grader",
        _grader_route,
        {
            "generate_response": "synthesis",
            "rewrite_query": "rewrite",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_conditional_edges(
        "synthesis",
        _after_synthesis_route,
        {
            "fallback": "fallback",
            "evaluator": "evaluator",
            "critique": "critique",
        },
    )
    workflow.add_conditional_edges(
        "evaluator",
        _route_after_eval,
        {
            "critique": "critique",
            "finish": END,
            "human_review": "human_review",
        },
    )
    workflow.add_conditional_edges(
        "critique",
        _should_continue,
        {
            "refine": "synthesis",
            "fallback": "fallback",
            "finish": END,
            "human_review": "human_review",
        },
    )
    workflow.add_edge("fallback", END)
    workflow.add_edge("human_review", END)

    return workflow.compile(checkpointer=checkpointer if checkpointer is not None else False)


def get_checkpoint_db_path() -> Path:
    """Path to the SQLite checkpoint database for persistent memory."""
    return Path(__file__).resolve().parents[2] / "checkpoints.db"
