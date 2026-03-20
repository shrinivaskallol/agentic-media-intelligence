"""
Unified Context Retriever: Two-stage retrieval (Postgres vector + Neo4j graph).
Provides standalone get_graph_context(entities) and get_vector_context(query) for entity workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psycopg2
from neo4j.exceptions import DriverError, ServiceUnavailable

from app.tools.db_utils import connect_postgres, get_neo4j_driver

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default embedding model (384 dims, matches pgvector)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class VectorResult:
    """A single chunk from Postgres vector search."""

    chunk_id: str
    content: str
    article_id: str
    chunk_index: int  # 0 if not in schema (derived from ROW_NUMBER when available)


@dataclass
class GraphResult:
    """A single triple from Neo4j graph."""

    head_name: str
    relation_type: str
    tail_name: str
    chunk_id: str | None


class UnifiedRetriever:
    """
    Two-stage retriever:
    Stage A: Vector search in Postgres (top K chunks)
    Stage B: Graph expansion in Neo4j (direct relations + neighbors of neighbors)
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_limit: int = 5,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.embedder: SentenceTransformer | None = None
        self.vector_limit = vector_limit

    def _get_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_model_name)
        return self.embedder

    def _vector_search(self, query: str, limit: int) -> tuple[list[VectorResult], list[str]]:
        """Stage A: Search Postgres for top K similar chunks. Returns (chunks, chunk_ids)."""
        chunks: list[VectorResult] = []
        chunk_ids: list[str] = []
        try:
            conn = connect_postgres()
            if conn is None:
                logger.warning("Postgres connection failed; skipping vector search")
                return chunks, chunk_ids
            try:
                embedder = self._get_embedder()
                query_vector = embedder.encode(query, convert_to_numpy=True)
                vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT chunk_hash, content, article_id
                        FROM article_chunks
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vec_str, limit),
                    )
                    rows = cur.fetchall()

                for row in rows:
                    chunk_hash, content, article_id = row
                    chunks.append(
                        VectorResult(
                            chunk_id=str(chunk_hash),
                            content=content or "",
                            article_id=str(article_id) if article_id is not None else "",
                            chunk_index=0,  # Not in schema; use 0 as placeholder
                        )
                    )
                    chunk_ids.append(str(chunk_hash))
            finally:
                conn.close()
        except (psycopg2.OperationalError, psycopg2.Error, OSError) as e:
            logger.warning("Vector search failed: %s", e)

        return chunks, chunk_ids

    def _graph_expansion(self, chunks: list[VectorResult]) -> list[GraphResult]:
        """Stage B: Query Neo4j using chunk_ids AND keyword fallbacks."""
        triples: list[GraphResult] = []
        seen: set[tuple[str, str, str]] = set()

        chunk_ids = [c.chunk_id for c in chunks]
        if not chunk_ids:
            return triples

        logger.info("Expanding graph for chunk_ids: %s", chunk_ids)

        try:
            driver = get_neo4j_driver()
            if driver is None:
                logger.warning("Neo4j driver failed; skipping graph expansion")
                return triples

            with driver.session() as session:
                # 1. ATTEMPT DIRECT ID MATCH
                direct_query = """
                MATCH (h)-[r]->(t)
                WHERE r.chunk_id IN $chunk_ids
                RETURN h.name AS head_name, type(r) AS rel_type, t.name AS tail_name, r.chunk_id AS chunk_id
                """
                result = session.run(direct_query, chunk_ids=chunk_ids)
                for rec in result:
                    self._add_to_triples(rec, triples, seen)

                # 2. FALLBACK: ENTITY-NAME MATCH (If ID match is empty or weak)
                if len(triples) < 3:
                    logger.info("Direct ID match weak; attempting keyword-based expansion...")
                    all_text = " ".join(c.content for c in chunks[:2])
                    fallback_query = """
                    MATCH (h)-[r]->(t)
                    WHERE any(name IN [h.name, t.name] WHERE name IS NOT NULL AND $text CONTAINS name)
                    RETURN h.name AS head_name, type(r) AS rel_type, t.name AS tail_name, r.chunk_id AS chunk_id
                    LIMIT 15
                    """
                    result_fb = session.run(fallback_query, text=all_text)
                    for rec in result_fb:
                        self._add_to_triples(rec, triples, seen)

            driver.close()
        except (ServiceUnavailable, DriverError, OSError) as e:
            logger.warning("Graph expansion failed: %s", e)

        return triples

    def _add_to_triples(
        self,
        rec,
        triples: list[GraphResult],
        seen: set[tuple[str, str, str]],
    ) -> None:
        """Helper to deduplicate and add results."""
        h = rec.get("head_name") or ""
        rel = rec.get("rel_type") or "RELATES_TO"
        t = rec.get("tail_name") or ""
        cid = rec.get("chunk_id")
        key = (h, rel, t)
        if key not in seen and h and t:
            seen.add(key)
            triples.append(GraphResult(head_name=h, relation_type=rel, tail_name=t, chunk_id=cid))

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Two-stage retrieval, then merge into a single context string.
        If one DB is down, still returns results from the other.
        """
        k = limit or self.vector_limit

        chunks, _chunk_ids = self._vector_search(query, k)
        triples = self._graph_expansion(chunks)

        parts: list[str] = []

        parts.append("--- TEXT CHUNKS ---")
        if chunks:
            for i, c in enumerate(chunks, 1):
                parts.append(f"[{i}] (chunk_id={c.chunk_id})")
                parts.append(c.content)
                parts.append("")
        else:
            parts.append("(No text chunks returned)")
            parts.append("")

        parts.append("--- GRAPH KNOWLEDGE ---")
        if triples:
            for t in triples:
                cid_str = f" [chunk_id={t.chunk_id}]" if t.chunk_id else ""
                parts.append(f"({t.head_name})-[{t.relation_type}]->({t.tail_name}){cid_str}")
        else:
            parts.append("(No graph triples returned)")
            parts.append("")

        return "\n".join(parts).strip()

    def retrieve_raw(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[list[VectorResult], list[GraphResult]]:
        """Retrieve without formatting; returns (chunks, triples) for testing."""
        k = limit or self.vector_limit
        chunks, _chunk_ids = self._vector_search(query, k)
        triples = self._graph_expansion(chunks)
        return chunks, triples


# --- Standalone functions for entity workflow (extract → retrieve) ---

_default_retriever: UnifiedRetriever | None = None
_default_embedder: SentenceTransformer | None = None


def _get_embedder():
    """Lazy-load sentence-transformers for vector generation (only when vector search runs)."""
    global _default_embedder
    if _default_embedder is None:
        from sentence_transformers import SentenceTransformer

        _default_embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _default_embedder


def get_graph_context(entities: list[str], limit: int = 15) -> list[str]:
    """Delegates to app.logic.retrieval for multi-hop graph context."""
    from app.logic.retrieval import get_graph_context as _get_graph_context

    return _get_graph_context(entities, limit)


def _detect_article_chunks_schema(cur) -> str:
    """Return 'content' (01) or 'entity' (02) based on actual table columns."""
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'article_chunks'
        """
    )
    cols = {r[0] for r in cur.fetchall()}
    if "content" in cols:
        return "content"  # 01 schema: chunk_hash, content, article_id
    if "entity" in cols:
        return "entity"  # 02 schema: chunk_id, entity, entity_type, confidence
    return "content"  # default fallback


def get_vector_context(
    query_text: str,
    query_vector: list[float] | None = None,
    limit: int = 5,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Finds semantically relevant text chunks from Postgres.
    Returns (context_bits, chunk_ids, metadata) for structured context with entity types.
    metadata: list of {"entity": str, "entity_type": str} for [ENTITY_TYPE] entity: content format.
    """
    from app.tools.db_utils import get_pg_conn

    if query_vector is None:
        embedder = _get_embedder()
        query_vector = embedder.encode(query_text, convert_to_numpy=True).tolist()

    vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    context_bits: list[str] = []
    chunk_ids: list[str] = []
    metadata: list[dict] = []

    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            schema = _detect_article_chunks_schema(cur)
            if schema == "entity":
                cur.execute(
                    """
                    SELECT entity, entity_type, confidence
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_str, limit),
                )
                rows = cur.fetchall()
                for row in rows:
                    entity, entity_type, confidence = row[0], row[1], row[2]
                    etype = str(entity_type).upper() if entity_type else "UNKNOWN"
                    content = f"Entity: {entity} ({entity_type}) with confidence {confidence}"
                    context_bits.append(content)
                    chunk_ids.append(str(entity))
                    metadata.append(
                        {"entity": str(entity) if entity else None, "entity_type": etype}
                    )
            else:
                # 01 schema: chunk_hash, content, article_id
                cur.execute(
                    """
                    SELECT chunk_hash, content, article_id
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_str, limit),
                )
                rows = cur.fetchall()
                for i, row in enumerate(rows, 1):
                    chunk_hash, content, _ = row[0], row[1], row[2]
                    chunk_ids.append(str(chunk_hash))
                    context_bits.append(f"[{i}] {content or ''}")
                    metadata.append({"entity": None, "entity_type": "TEXT"})
        conn.close()
    except (psycopg2.OperationalError, psycopg2.Error, OSError) as e:
        logger.warning("get_vector_context failed: %s", e)

    return context_bits, chunk_ids, metadata


def _get_default_retriever() -> UnifiedRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = UnifiedRetriever()
    return _default_retriever
