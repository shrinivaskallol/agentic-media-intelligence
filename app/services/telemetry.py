"""
Telemetry for graph execution metrics. Refusal rate, revisions, and final score.
Industry standard: log to stdout, CSV, SQLite, or Prometheus.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_graph_metrics(final_state: dict) -> None:
    """
    Log refusal and quality metrics after graph execution.
    Use for Refusal Rate dashboards and retrieval quality monitoring.
    """
    metrics = {
        "refused": final_state.get("is_refused", False),
        "revisions": final_state.get("revision_count", 0),
        "final_score": final_state.get("critique_score", 0.0),
        "query": final_state.get("query", "")[:100],  # Truncate for log
    }
    logger.info("METRICS CAPTURED: %s", metrics)
