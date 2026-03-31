"""
CLI entry points for scripts. Use: uv run ami-workflow, uv run ami-init-db, etc.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on path when running as installed script
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")


def _state_has_interrupt(result: dict) -> bool:
    intr = result.get("__interrupt__")
    if intr is None:
        return False
    if isinstance(intr, (list, tuple)):
        return len(intr) > 0
    return bool(intr)


def run_workflow() -> None:
    """Run the CRAG workflow demo. Optional query from argv: ``uv run ami-workflow \"your question\"``."""
    from app import configure_logging
    from app.graph.entity_workflow import build_workflow
    from app.logic.workflow import make_initial_state

    default_query = (
        "Identify the primary optics provider for Apple's 3nm chip manufacturer."
    )
    argv = sys.argv[1:]
    query = " ".join(argv).strip() if argv else default_query

    async def _run() -> None:
        configure_logging()
        app = build_workflow()
        inputs = make_initial_state(query)
        result = await app.ainvoke(inputs)
        result = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
        import logging

        from app.services.telemetry import log_workflow_outcome

        log_workflow_outcome(
            result,
            hitl_interrupted=_state_has_interrupt(result),
        )

        logger = logging.getLogger(__name__)
        logger.info(
            "Final revision_count=%s retrieval_revision_count=%s is_passing=%s",
            result.get("revision_count", 0),
            result.get("retrieval_revision_count", 0),
            result.get("ragas_scores", {}).get("is_passing", "N/A"),
        )
        interrupted = _state_has_interrupt(result)
        if interrupted:
            logger.info(
                "Graph paused for HITL (diversity / MMR λ). No synthesis yet — resume via MCP "
                "or Streamlit, or set AMI_HITL_DIVERSITY=0 for a non-interactive CLI run."
            )
        elif result.get("retrieval_revision_count", 0) > 0:
            logger.info("SUCCESS: Agentic loop triggered and self-corrected.")
        else:
            logger.info("NOTE: System found answer in one hit (no loop needed).")

        # Human-readable summary for smoke tests (stderr log + stdout)
        exit_reason = str(result.get("exit_reason") or "").strip()
        partial = bool(result.get("partial_answer"))
        print()
        print("--- ami-workflow result ---")
        print(f"query: {query[:200]}{'…' if len(query) > 200 else ''}")
        print(f"exit_reason: {exit_reason or '(none)'}")
        print(f"partial_answer: {partial}")
        print(f"retrieval_revision_count: {result.get('retrieval_revision_count', 0)}")
        if interrupted:
            print("status: interrupted (HITL — diversity / MMR λ)")
            print(
                "hint: Synthesis did not run. For a full answer from the CLI, unset or set "
                "AMI_HITL_DIVERSITY=0 in .env. Or resume with the same thread_id via MCP/dashboard."
            )
        resp = (result.get("response") or "").strip()
        preview = resp[:1200] + ("…" if len(resp) > 1200 else "")
        print("response:")
        if interrupted and not preview:
            print(
                "(empty — expected: graph stopped at diversity interrupt before grader/synthesis)"
            )
        else:
            print(preview or "(empty)")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def init_db() -> None:
    """Seed Postgres, Neo4j, Redis infrastructure."""
    from app import configure_logging

    configure_logging()
    from scripts.init_db import main

    main()


def migrate() -> None:
    """Run Alembic migrations (Postgres schema)."""
    from app import configure_logging

    configure_logging()
    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config(Path(__file__).resolve().parents[1] / "alembic.ini")
    command.upgrade(alembic_cfg, "head")


def seed_postgres() -> None:
    """Seed synthetic Postgres vector data."""
    from scripts.seed_synthetic_postgres import main

    main()


def seed_neo4j() -> None:
    """Seed synthetic Neo4j graph data."""
    from scripts.seed_synthetic_neo4j import run_golden_graph_ingestion

    run_golden_graph_ingestion()
