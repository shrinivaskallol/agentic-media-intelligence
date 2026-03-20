"""
Suppress noisy asyncio/GenAI shutdown messages:
- "Task was destroyed but it is pending!"
- RuntimeWarning: coroutine 'BaseApiClient.aclose' was never awaited

Call install_suppress_async_noise() at process start (e.g. in tests/conftest.py, scripts/run_workflow.py).
"""

from __future__ import annotations

import sys
import warnings


class _FilteredStderr:
    """Stderr wrapper that drops noisy async shutdown lines."""

    _DROP_PATTERNS = (
        "Task was destroyed but it is pending",
        "task: <Task pending",
        "BaseApiClient.aclose",
    )

    def __init__(self, stream):
        self._stream = stream

    def write(self, s: str) -> int:
        if s and any(p in s for p in self._DROP_PATTERNS):
            return 0
        return self._stream.write(s)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_suppress_async_noise() -> None:
    """Suppress RuntimeWarning and filter stderr for async cleanup noise."""
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*coroutine.*was never awaited",
    )
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)
