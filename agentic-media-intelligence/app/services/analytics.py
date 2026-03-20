"""
Analytics DB for Refusal Rate and query quality over time.
Logs metrics to SQLite for easy plotting and dashboards.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path

# DB path: project root (next to checkpoints.db)
_DB_PATH = Path(__file__).resolve().parents[2] / "agent_metrics.db"


def log_event(query: str, state: dict) -> None:
    """
    Log a single graph execution to the analytics DB.
    Non-blocking for minimal impact on agent latency.
    """
    conn = sqlite3.connect(str(_DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            is_refused INTEGER,
            score REAL,
            revisions INTEGER
        )
        """
    )
    cur.execute(
        "INSERT INTO metrics (timestamp, query, is_refused, score, revisions) VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            (query or "")[:500],  # Truncate for storage
            1 if state.get("is_refused") else 0,
            float(state.get("critique_score", 0.0)),
            int(state.get("revision_count", 0)),
        ),
    )
    conn.commit()
    conn.close()


def print_stats() -> None:
    """
    Print Refusal Rate and summary stats from the metrics DB.
    (Total Refusals / Total Queries) * 100 = Refusal Rate %
    """
    if not _DB_PATH.exists():
        logger.info("No metrics DB yet. Run some queries first.")
        return

    conn = sqlite3.connect(str(_DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(is_refused) AS refused,
            ROUND(AVG(score), 2) AS avg_score,
            ROUND(AVG(revisions), 1) AS avg_revisions
        FROM metrics
        """
    )
    row = cur.fetchone()
    conn.close()

    if not row or row[0] == 0:
        logger.info("No rows in metrics table.")
        return

    total, refused, avg_score, avg_revisions = row
    refusal_rate = (refused / total * 100) if total else 0.0

    logger.info(
        "REFUSAL TELEMETRY: total=%d refused=%d rate=%.1f%% avg_score=%s avg_revisions=%s",
        total,
        refused,
        refusal_rate,
        avg_score,
        avg_revisions,
    )
