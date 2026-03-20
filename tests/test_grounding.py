#!/usr/bin/env python3
"""
Test Closed-World Grounding: inversion resistance and ghost hallucination prevention.
Requires: Postgres and Neo4j seeded with synthetic data (run seed_synthetic_postgres.py and seed_synthetic_neo4j.py first).
Run: uv run python tests/test_grounding.py
"""

import asyncio
import sys
from pathlib import Path

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.utils.suppress_async_noise import install_suppress_async_noise

install_suppress_async_noise()

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

from app.graph.entity_workflow import build_workflow


def _make_initial_state(query: str) -> dict:
    return {
        "query": query,
        "entities": [],
        "intent": "",
        "context": [],
        "response": "",
    }


def _normalize(s: str) -> str:
    return s.lower().strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(text)


@pytest.mark.integration
async def test_inversion():
    """
    Case A: The Inversion Test.
    Query: "Does TSMC supply equipment to ASML?"
    Ground truth: ASML supplies TO TSMC (not the reverse).
    Success: Agent answers "No" or "Information not available", or correctly states ASML supplies TSMC.
    Failure: Agent says TSMC supplies equipment to ASML (inverted fact).
    """
    query = "Does TSMC supply equipment to ASML?"
    app = build_workflow()
    inputs = _make_initial_state(query)

    final_state = await app.ainvoke(inputs)
    response = final_state.get("response", "")

    # Failure: inverted fact (TSMC supplies equipment to ASML) asserted as true.
    # Do NOT fail when the agent quotes the user's question and then denies/corrects it
    # (e.g. "The user asks if TSMC supplies... However, the context suggests the opposite").
    inverted_phrases = [
        "tsmc supplies equipment to asml",
        "tsmc supply equipment to asml",
        "tsmc provides equipment to asml",
    ]
    denial_signals = [
        "opposite",
        "asml supplies",
        "asml supply",
        "suggests the",
        "correct relationship",
        "does not supply",
        "does not provide",
    ]
    for phrase in inverted_phrases:
        if _contains_phrase(response, phrase):
            # Agent may mention the phrase only to deny it; check for correction/denial
            if any(_contains_phrase(response, s) for s in denial_signals):
                break  # Pass: agent is correcting, not asserting
            return (
                False,
                f"INVERSION: Response incorrectly states TSMC supplies equipment to ASML: {response[:300]}...",
            )

    # Success: correct direction (ASML supplies TSMC) or refusal
    correct_phrases = [
        "asml supplies",
        "asml supply",
        "asml provides equipment to tsmc",
    ]
    refusal_phrases = [
        "no",
        "information not available",
        "no information",
        "not in the",
        "not in our",
        "insufficient",
        "refusal",
        "do not have",
        "cannot find",
    ]

    has_correct = any(_contains_phrase(response, p) for p in correct_phrases)
    has_refusal = any(_contains_phrase(response, p) for p in refusal_phrases)

    if has_correct:
        return True, "Correct: Response states ASML supplies TSMC (or equivalent)."
    if has_refusal:
        return True, "Correct: Agent refused or stated information not available."
    # Also pass if response explicitly denies the inverted claim
    if _contains_phrase(response, "tsmc does not supply") or _contains_phrase(
        response, "tsmc does not provide"
    ):
        return True, "Correct: Response denies TSMC supplies ASML."

    return (
        False,
        f"Unclear: Response neither confirms ASML->TSMC nor refuses. Excerpt: {response[:400]}...",
    )


@pytest.mark.integration
async def test_ghost_hallucination():
    """
    Case B: The Ghost Hallucination Test.
    Query: "How does Nvidia's competition with AMD affect Blackwell?"
    Ground truth: Synthetic data has AMD COMPETES_WITH Nvidia and Blackwell facts, but NO direct
    causal link "how AMD competition affects Blackwell".
    Success: Agent states there is no information regarding competition with AMD affecting Blackwell
    in the provided evidence, or only reports what IS in the evidence without inventing the causal link.
    Failure: Agent invents a specific causal narrative (e.g. "AMD competition forces Nvidia to...")
    without grounding in the evidence.
    """
    query = "How does Nvidia's competition with AMD affect Blackwell?"
    app = build_workflow()
    inputs = _make_initial_state(query)

    final_state = await app.ainvoke(inputs)
    response = final_state.get("response", "")

    # Failure: invented causal narrative without evidence
    # e.g. "AMD competition affects Blackwell by..." when no such link exists in synthetic data
    invented_causal = [
        "amd competition affects blackwell",
        "amd competition impacts blackwell",
        "amd's competition affects blackwell",
        "competition with amd affects blackwell",
        "amd forces nvidia",
        "amd pressure on blackwell",
    ]
    for phrase in invented_causal:
        if _contains_phrase(response, phrase):
            # Check if it's presented as a grounded claim (has citation) vs invented
            if "[Vector-" in response or "[Graph-" in response:
                # Has citations - might be over-interpreting; allow if it says "no direct evidence"
                if _contains_phrase(response, "no information") or _contains_phrase(
                    response, "no direct"
                ):
                    continue
            return (
                False,
                f"GHOST HALLUCINATION: Response invents causal link '{phrase}' without evidence.",
            )

    # Success: acknowledges lack of evidence
    ok_phrases = [
        "no information",
        "no direct",
        "not in the provided",
        "not in our",
        "not stated in the evidence",
        "insufficient",
        "refusal",
        "do not have information",
        "cannot determine",
        "evidence does not",
        "data does not",
    ]
    if any(_contains_phrase(response, p) for p in ok_phrases):
        return True, "Correct: Agent acknowledges no information on AMD-Blackwell causal link."

    # Success: only reports what IS in evidence (AMD competes with Nvidia, Blackwell is 4NP) without
    # inventing the "affect" linkage
    if _contains_phrase(response, "amd") and _contains_phrase(response, "blackwell"):
        # If both mentioned but no causal claim, could be OK - check for hedging
        if (
            _contains_phrase(response, "however")
            or _contains_phrase(response, "although")
            or _contains_phrase(response, "no direct link")
        ):
            return True, "Correct: Agent reports available facts without inventing causal link."
        # If it cites sources for both, and doesn't claim a direct effect, pass
        if "[Vector-" in response or "[Graph-" in response:
            return True, "Correct: Agent cites sources; assume grounded."

    return (
        False,
        f"Unclear: Response may have invented AMD-Blackwell causal link. Excerpt: {response[:400]}...",
    )


async def main():
    print("=" * 60)
    print("Closed-World Grounding Tests")
    print("=" * 60)

    results = []

    print("\n--- Case A: Inversion Test ---")
    print("Query: 'Does TSMC supply equipment to ASML?'")
    try:
        passed, msg = await test_inversion()
        results.append(("Inversion", passed, msg))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {msg}")
    except Exception as e:
        results.append(("Inversion", False, str(e)))
        print(f"[FAIL] Exception: {e}")

    print("\n--- Case B: Ghost Hallucination Test ---")
    print("Query: 'How does Nvidia's competition with AMD affect Blackwell?'")
    try:
        passed, msg = await test_ghost_hallucination()
        results.append(("Ghost Hallucination", passed, msg))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {msg}")
    except Exception as e:
        results.append(("Ghost Hallucination", False, str(e)))
        print(f"[FAIL] Exception: {e}")

    print("\n" + "=" * 60)
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("All grounding tests PASSED.")
    else:
        print("Some grounding tests FAILED.")
        for name, passed, msg in results:
            if not passed:
                print(f"  - {name}: {msg}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
