"""
Graph retrieval logic with multi-hop discovery.
Supports variable-length paths (1..3 hops) for Tier-3 supply chain traversal.
Returns triples with full entity property maps (headquarters, region) for LLM grounding.
"""

import logging
import re
from typing import Any, List

logger = logging.getLogger(__name__)


def _parse_graph_results(records: list[dict[str, Any]]) -> List[str]:
    """
    Format Neo4j records into high-signal strings for the LLM.
    Extracts headquarters and region when available; falls back to N/A.
    """
    formatted: list[str] = []
    seen: set[str] = set()
    for rec in records:
        s_props = rec.get("source_props") or {}
        t_props = rec.get("target_props") or {}
        s_name = rec.get("source_name")
        t_name = rec.get("target_name")
        if not s_name or not t_name:
            continue
        rel_type = rec.get("relationship_type") or "RELATED_TO"
        cid = (rec.get("rel_props") or {}).get("chunk_id", "")

        s_loc = f" ({s_props.get('headquarters', 'N/A')}, {s_props.get('region', 'N/A')})"
        t_loc = f" ({t_props.get('headquarters', 'N/A')}, {t_props.get('region', 'N/A')})"

        line = f"[{s_name}{s_loc}] --{rel_type}--> [{t_name}{t_loc}]"
        if line not in seen:
            seen.add(line)
            if cid:
                line += f" [Graph-{cid}]"
            formatted.append(line)
    return formatted


def get_graph_context(entities: List[str], limit: int = 25) -> List[str]:
    """
    Expands context by finding neighbors of extracted entities in Neo4j.
    Uses variable-length paths (1..3 hops) for Tier-3 supply chain traversal.
    Returns distinct triples with entity metadata (headquarters, region) when available.
    Handles entities identified by id or name.
    """
    if not entities:
        return []

    context_bits: list[str] = []
    try:
        from app.tools.db_utils import get_neo4j_driver

        driver = get_neo4j_driver()
        simple_ids = [e.lower().replace(" ", "_") for e in entities]
        canonical_ids = [re.sub(r"[\s\-_]", "", e.lower()) for e in entities]
        composite_ids = [
            f"ORGANIZATION_{e.lower().replace(' ', '_')}" for e in entities
        ] + [f"PRODUCT_{e.lower().replace(' ', '_')}" for e in entities]
        ids = list(dict.fromkeys(simple_ids + canonical_ids + composite_ids))

        query = """
        MATCH (startNode:Entity)
        WHERE startNode.id IN $ids OR startNode.name IN $names
        MATCH path = (startNode)-[r*1..3]-(neighbor:Entity)
        UNWIND r AS rel
        WITH DISTINCT rel
        RETURN
            startNode(rel).name AS source_name,
            properties(startNode(rel)) AS source_props,
            type(rel) AS relationship_type,
            properties(rel) AS rel_props,
            endNode(rel).name AS target_name,
            properties(endNode(rel)) AS target_props
        LIMIT $limit
        """
        with driver.session() as session:
            result = session.run(
                query,
                ids=ids,
                names=entities,
                limit=limit,
            )
            records = [dict(rec) for rec in result]
        driver.close()

        context_bits = _parse_graph_results(records)
    except Exception as e:
        logger.warning("get_graph_context failed: %s", e)

    return context_bits
