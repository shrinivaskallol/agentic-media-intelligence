"""Pre/post synthesis hooks: closed-world gate and citation coverage."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.policies.config import MIN_CITATION_LINE_RATIO
from app.policies.evidence import meets_min_evidence
from app.utils.citations import build_numbered_context_for_llm

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[\d+\]")


def _state_flag(state: Any, name: str, default: bool = False) -> bool:
    val = getattr(state, name, None)
    if val is None and isinstance(state, dict):
        val = state.get(name, default)
    return bool(val)


def _exit_reason(state: Any) -> str:
    er = getattr(state, "exit_reason", None)
    if er is None and isinstance(state, dict):
        er = state.get("exit_reason", "")
    return str(er or "").strip()


def check_closed_world_before_synthesis(state: Any) -> dict | None:
    """
    Pre-synthesis: block LLM call when there is no indexed evidence unless partial/no-evidence path.
    """
    if _state_flag(state, "partial_answer"):
        return None
    if _exit_reason(state) == "max_retries_exceeded":
        return None

    numbered = build_numbered_context_for_llm(state).strip()
    no_numbered = (not numbered) or numbered == "No relevant data retrieved."
    if not no_numbered and meets_min_evidence(state):
        return None

    if no_numbered and not meets_min_evidence(state):
        logger.info("POLICY PreSynthesis: closed-world block — no grounded evidence")
        return {
            "response": (
                "## No evidence retrieved\n\n"
                "The closed-world policy blocked report generation: no grounded evidence "
                "is available in the knowledge base for this query.\n\n"
                "(Deterministic policy: insufficient indexed evidence.)"
            ),
            "is_refused": True,
            "synthesis_status": "no_evidence",
            "synthesis_confidence": 0.0,
            "synthesis_requires_critique": False,
        }

    return None


def check_citation_coverage(
    report_markdown: str,
    *,
    min_ratio: float | None = None,
) -> tuple[bool, str]:
    """
    Post-synthesis: require bracket citations [n] on a minimum share of substantive lines.
    Returns (passed, reason_if_failed).
    """
    threshold = MIN_CITATION_LINE_RATIO if min_ratio is None else min_ratio
    text = (report_markdown or "").strip()
    if not text:
        return False, "empty_report"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    substantive = [
        ln
        for ln in lines
        if len(ln) > 50
        and not ln.startswith("#")
        and not ln.startswith("|")
        and not ln.startswith("- ")
        and not ln.startswith("* ")
    ]
    if not substantive:
        return True, ""

    cited = sum(1 for ln in substantive if _CITATION.search(ln))
    ratio = cited / len(substantive)
    if ratio < threshold:
        return False, f"citation_line_ratio={ratio:.2f}<{threshold:.2f}"
    return True, ""
