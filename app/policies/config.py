"""Policy thresholds — env overrides with safe defaults."""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Retrieval budget (PreToolUse)
MAX_ENTITIES = _int_env("AMI_MAX_ENTITIES", 5)
MAX_GRAPH_LIMIT = _int_env("AMI_MAX_GRAPH_LIMIT", 5)
MAX_VECTOR_LIMIT = _int_env("AMI_MAX_VECTOR_LIMIT", 3)

# Retrieval loop circuit breaker
MAX_RETRIEVAL_REVISIONS = _int_env("AMI_MAX_RETRIEVAL_REVISIONS", 3)

# Min evidence (PostToolUse / grader)
MIN_GRAPH_FACTS = _int_env("AMI_MIN_GRAPH_FACTS", 1)
MIN_VECTOR_CHUNKS = _int_env("AMI_MIN_VECTOR_CHUNKS", 2)

# Synthesis citation coverage (PostToolUse)
MIN_CITATION_LINE_RATIO = _float_env("AMI_MIN_CITATION_LINE_RATIO", 0.25)

# Quality / escalation
MAX_REFINEMENT_REVISIONS = _int_env("AMI_MAX_REFINEMENT_REVISIONS", 3)
MAX_SYNTHESIS_SELF_CORRECTION = _int_env("AMI_MAX_SYNTHESIS_SELF_CORRECTION", 2)
