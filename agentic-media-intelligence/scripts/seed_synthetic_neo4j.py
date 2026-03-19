#!/usr/bin/env python3
"""
Wipe Neo4j graph, seed with synthetic golden nodes and relationships.
Schema matches 02-prepare-graph-neo4j-data: Entity nodes, relationship types, chunk_id on rels.
Run: uv run python scripts/seed_synthetic_neo4j.py
Requires: Neo4j with APOC plugin (for apoc.merge.relationship).
"""

import os
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj))
os.chdir(_proj)

from dotenv import load_dotenv

load_dotenv(_proj / ".env")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
PWD = os.getenv("NEO4J_PASSWORD", "password")

NODE_DATA = [
    {"id": "nvidia", "name": "Nvidia", "type": "ORGANIZATION"},
    {"id": "apple", "name": "Apple", "type": "ORGANIZATION"},
    {"id": "microsoft", "name": "Microsoft", "type": "ORGANIZATION"},
    {"id": "tesla", "name": "Tesla", "type": "ORGANIZATION"},
    {"id": "google", "name": "Google", "type": "ORGANIZATION"},
    {"id": "amazon", "name": "Amazon", "type": "ORGANIZATION"},
    {"id": "meta", "name": "Meta", "type": "ORGANIZATION"},
    {"id": "amd", "name": "AMD", "type": "ORGANIZATION"},
    {"id": "intel", "name": "Intel", "type": "ORGANIZATION"},
    {"id": "openai", "name": "OpenAI", "type": "ORGANIZATION"},
    {"id": "qualcomm", "name": "Qualcomm", "type": "ORGANIZATION"},
    {"id": "tsmc", "name": "TSMC", "type": "ORGANIZATION"},
    {"id": "samsung", "name": "Samsung", "type": "ORGANIZATION"},
    {"id": "asml", "name": "ASML", "type": "ORGANIZATION"},
    {"id": "zeiss", "name": "Carl Zeiss AG", "type": "ORGANIZATION"},
    {"id": "anthropic", "name": "Anthropic", "type": "ORGANIZATION"},
    {"id": "blackwell", "name": "Blackwell", "type": "ARCHITECTURE"},
    {"id": "stargate", "name": "Stargate", "type": "PROJECT"},
    {"id": "cuda", "name": "CUDA", "type": "TECHNOLOGY"},
    {"id": "hbm3e", "name": "HBM3e", "type": "TECHNOLOGY"},
    {"id": "cowos", "name": "CoWoS", "type": "TECHNOLOGY"},
    # Toxic seed: fake entities to test RAG grounding (if agent mentions them, RAG works; if it defaults to TSMC/Nvidia, grounding leaks)
    {"id": "xylos6", "name": "Xylos-6", "type": "ORGANIZATION"},
    {"id": "zenthcore", "name": "Zenth-Core", "type": "ORGANIZATION"},
]

REL_DATA = [
    # Supply Chain (direction: h -> t, e.g. ASML supplies TO TSMC)
    {"h_id": "nvidia", "t_id": "tsmc", "rel": "PARTNERS_WITH", "chunk_id": "syn_001"},
    {"h_id": "blackwell", "t_id": "tsmc", "rel": "MANUFACTURED_BY", "chunk_id": "syn_002"},
    {"h_id": "asml", "t_id": "tsmc", "rel": "SUPPLIES_EQUIPMENT_TO", "chunk_id": "syn_003"},  # ASML->TSMC (not TSMC->ASML)
    {"h_id": "zeiss", "t_id": "asml", "rel": "SUPPLIES_OPTICS_TO", "chunk_id": "syn_013"},  # Zeiss->ASML->TSMC->Apple (3-hop chain)
    {"h_id": "samsung", "t_id": "nvidia", "rel": "SUPPLIES_HBM_TO", "chunk_id": "syn_004"},
    # Competition & Partnerships
    {"h_id": "microsoft", "t_id": "openai", "rel": "INVESTED_IN", "chunk_id": "syn_005"},
    {"h_id": "microsoft", "t_id": "stargate", "rel": "DEVELOPING", "chunk_id": "syn_006"},
    {"h_id": "amd", "t_id": "nvidia", "rel": "COMPETES_WITH", "chunk_id": "syn_007"},
    {"h_id": "meta", "t_id": "nvidia", "rel": "BUYING_CHIPS_FROM", "chunk_id": "syn_008"},
    # Technology
    {"h_id": "nvidia", "t_id": "cuda", "rel": "DEVELOPED", "chunk_id": "syn_009"},
    {"h_id": "apple", "t_id": "tsmc", "rel": "SECURED_3NM_CAPACITY", "chunk_id": "syn_010"},
    {"h_id": "tesla", "t_id": "nvidia", "rel": "USES_HARDWARE_FROM", "chunk_id": "syn_011"},
    {"h_id": "google", "t_id": "anthropic", "rel": "INVESTED_IN", "chunk_id": "syn_012"},
    # Toxic seed: Xylos-6 supplies plasma channels to Zenth-Core (fake relationship for grounding test)
    {"h_id": "xylos6", "t_id": "zenthcore", "rel": "SUPPLIES_PLASMA_CHANNELS_TO", "chunk_id": "tox_001"},
]


def run_golden_graph_ingestion():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(URI, auth=(USER, PWD))

    node_query = """
    UNWIND $batch AS row
    MERGE (e:Entity {id: row.id})
    SET e.name = row.name, e.type = row.type
    """

    rel_query = """
    UNWIND $batch AS row
    MATCH (h:Entity {id: row.h_id})
    MATCH (t:Entity {id: row.t_id})
    CALL apoc.merge.relationship(h, row.rel, {}, {chunk_id: row.chunk_id}, t) YIELD rel
    RETURN count(*)
    """

    with driver.session() as session:
        print("Wiping graph...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Creating constraint...")
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )

        print("Upserting golden nodes...")
        session.run(node_query, batch=NODE_DATA)
        print(f"  {len(NODE_DATA)} nodes")

        print("Upserting golden relationships...")
        session.run(rel_query, batch=REL_DATA)
        print(f"  {len(REL_DATA)} relationships")

        # Toxic seed "Truth": Xylos-6 metadata for grounding test
        print("Enriching Xylos-6 (toxic seed)...")
        session.run("""
            MATCH (e:Entity {id: 'xylos6'})
            SET e.headquarters = 'Antwerp, Belgium',
                e.region = 'Europe',
                e.description = 'A European deep-tech firm specializing in post-silicon lithography.'
        """)

    driver.close()
    print("Golden graph ingestion complete.")


if __name__ == "__main__":
    run_golden_graph_ingestion()
