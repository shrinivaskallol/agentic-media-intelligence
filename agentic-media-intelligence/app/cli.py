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


def run_workflow() -> None:
    """Run the CRAG workflow demo."""
    from app import configure_logging
    from app.graph.entity_workflow import build_workflow
    from app.logic.workflow import make_initial_state

    async def _run() -> None:
        configure_logging()
        app = build_workflow()
        query = "Identify the primary optics provider for Apple's 3nm chip manufacturer."
        inputs = make_initial_state(query)
        result = await app.ainvoke(inputs)
        result = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "Final revision_count=%s retrieval_revision_count=%s is_passing=%s",
            result.get("revision_count", 0),
            result.get("retrieval_revision_count", 0),
            result.get("ragas_scores", {}).get("is_passing", "N/A"),
        )
        if result.get("retrieval_revision_count", 0) > 0:
            logger.info("SUCCESS: Agentic loop triggered and self-corrected.")
        else:
            logger.info("NOTE: System found answer in one hit (no loop needed).")

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
