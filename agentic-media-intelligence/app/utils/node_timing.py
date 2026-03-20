"""
Per-node latency instrumentation using time.perf_counter().
Wraps LangGraph nodes to log real execution time.
"""

import asyncio
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def timed_node(node_func: Callable, node_name: str) -> Callable:
    """
    Wraps a node function to log latency before and after execution.
    Supports both sync and async nodes.
    """

    async def _async_wrapped(state):
        t0 = time.perf_counter()
        try:
            return await node_func(state)
        finally:
            elapsed = time.perf_counter() - t0
            logger.info("LATENCY %s: %.3fs", node_name, elapsed)

    def _sync_wrapped(state):
        t0 = time.perf_counter()
        try:
            return node_func(state)
        finally:
            elapsed = time.perf_counter() - t0
            logger.info("LATENCY %s: %.3fs", node_name, elapsed)

    if asyncio.iscoroutinefunction(node_func):
        return _async_wrapped
    return _sync_wrapped
