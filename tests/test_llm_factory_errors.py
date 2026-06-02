"""Tests for LLM factory structured failures and retry backoff."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))

from app.errors.exceptions import LLMInvocationError
from app.llm_factory import FallbackLLM, _backoff_before_fallback, _is_fallback_error


def test_is_fallback_error_detects_rate_limit() -> None:
    assert _is_fallback_error(RuntimeError("429 Too Many Requests")) is True


def test_backoff_skips_non_retryable() -> None:
    with patch("app.llm_factory.time.sleep") as sl:
        _backoff_before_fallback(ValueError("bad schema"), 0)
        sl.assert_not_called()


def test_backoff_sleeps_for_retryable() -> None:
    with patch("app.llm_factory.time.sleep") as sl:
        _backoff_before_fallback(RuntimeError("429 rate limit"), 0)
        sl.assert_called_once()


def test_fallback_llm_raises_llm_invocation_error_on_hard_failure() -> None:
    bad = MagicMock()
    bad.invoke.side_effect = ValueError("invalid request payload")
    llm = FallbackLLM(candidates=[(bad, "test")])

    with pytest.raises(LLMInvocationError) as exc_info:
        llm.invoke("hello")

    assert exc_info.value.tool_error.error_category == "validation"
