"""
Graph-only retrieval: Neo4j entity expansion.
No sentence_transformers/torch — use for smoke tests when ML stack is broken.
Re-exports get_graph_context from app.logic.retrieval (multi-hop).
"""

from app.logic.retrieval import get_graph_context

__all__ = ["get_graph_context"]
