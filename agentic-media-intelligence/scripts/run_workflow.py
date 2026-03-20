#!/usr/bin/env python3
"""
Run the CRAG workflow demo with the "Apple Optics" query.
Requires a deep hop (Zeiss -> ASML -> TSMC -> Apple) that may trigger
grader -> rewrite -> re-retrieve loop.
Run: uv run python scripts/run_workflow.py
"""

import asyncio
import logging
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj))

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from app import configure_logging
from app.graph.entity_workflow import build_workflow
from app.logic.workflow import make_initial_state

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    app = build_workflow()
    query = "Identify the primary optics provider for Apple's 3nm chip manufacturer."

    logger.info("Executing Agentic Query: %s", query)
    inputs = make_initial_state(query)
    result = await app.ainvoke(inputs)

    if hasattr(result, "model_dump"):
        result = result.model_dump()
    else:
        result = dict(result) if result else {}

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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
