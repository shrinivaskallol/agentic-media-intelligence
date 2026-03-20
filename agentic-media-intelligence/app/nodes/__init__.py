"""Workflow nodes for entity extraction, retrieval, and synthesis."""

__all__ = [
    "entity_extractor",
    "extract_entities_node",
    "hybrid_retrieval_node",
    "synthesis_node",
]


def __getattr__(name: str):
    """Lazy imports to avoid pulling in heavy deps (transformers/numpy) when only evaluator is needed."""
    if name == "entity_extractor":
        from app.nodes.extraction import entity_extractor

        return entity_extractor
    if name == "extract_entities_node":
        from app.nodes.extraction import extract_entities_node

        return extract_entities_node
    if name == "hybrid_retrieval_node":
        from app.nodes.retriever import hybrid_retrieval_node

        return hybrid_retrieval_node
    if name == "synthesis_node":
        from app.nodes.synthesis import synthesis_node

        return synthesis_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
