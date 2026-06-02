"""
Unified Context Retriever: Two-stage retrieval (Postgres vector + Neo4j graph).
Provides standalone get_graph_context(entities) and get_vector_context(query) for entity workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import psycopg2
from neo4j.exceptions import DriverError, ServiceUnavailable

from app.errors.helpers import classify_exception, retrieval_slice_from_exception
from app.errors.models import RetrievalSlice
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


def get_graph_context(entities: list[str], limit: int = 15) -> RetrievalSlice:
    """Delegates to app.logic.retrieval for multi-hop graph context."""
    from app.logic.retrieval import get_graph_context as _get_graph_context

    return _get_graph_context(entities, limit)


def _vec_to_numpy(v) -> np.ndarray:
    """Parse pgvector / list / string repr into a 1-D float array."""
    if v is None:
        raise ValueError("null embedding")
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).ravel()
    if isinstance(v, (list, tuple, memoryview)):
        return np.asarray(v, dtype=np.float64).ravel()
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return np.array([float(x) for x in parts], dtype=np.float64)


def _mmr_order(
    query_vec: np.ndarray,
    doc_embs: np.ndarray,
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Maximal marginal relevance: return indices into doc_embs (rows) in selection order."""
    q = np.asarray(query_vec, dtype=np.float64).ravel()
    q = q / (np.linalg.norm(q) + 1e-9)
    d = np.asarray(doc_embs, dtype=np.float64)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    norms = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    d = d / norms
    n = d.shape[0]
    k = min(k, n)
    rel = d @ q
    selected: list[int] = []
    candidates = set(range(n))
    for _ in range(k):
        best_i = -1
        best_score = -np.inf
        for i in candidates:
            if not selected:
                score = lambda_mult * rel[i]
            else:
                sim_to_sel = float(np.max(d[i] @ d[selected].T))
                score = lambda_mult * rel[i] - (1.0 - lambda_mult) * sim_to_sel
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


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
    mmr_lambda: float = 1.0,
) -> tuple[list[str], list[str], list[dict], RetrievalSlice | None]:
    """
    Finds semantically relevant text chunks from Postgres.
    When mmr_lambda < 1.0, applies MMR on top of vector similarity (fetch_k > limit).

    Returns (context_bits, chunk_ids, metadata, error_slice).
    error_slice is set when Postgres/embedding failed; empty results with no error mean no_evidence.
    metadata: list of {"entity": str, "entity_type": str} for [ENTITY_TYPE] entity: content format.
    """
    from app.tools.db_utils import get_pg_conn

    embedder = _get_embedder()
    if query_vector is None:
        q_arr = embedder.encode(query_text, convert_to_numpy=True)
    else:
        q_arr = np.asarray(query_vector, dtype=np.float64).ravel()

    vec_str = "[" + ",".join(str(float(x)) for x in q_arr) + "]"
    context_bits: list[str] = []
    chunk_ids: list[str] = []
    metadata: list[dict] = []
    error_slice: RetrievalSlice | None = None

    use_mmr = mmr_lambda < 1.0 - 1e-9
    fetch_k = max(limit * 4, limit + 1) if use_mmr else limit

    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            schema = _detect_article_chunks_schema(cur)
            if schema == "entity":
                sql = (
                    """
                    SELECT entity, entity_type, confidence, embedding
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    if use_mmr
                    else """
                    SELECT entity, entity_type, confidence
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                )
                cur.execute(sql, (vec_str, fetch_k))
                rows = cur.fetchall()
                if not rows:
                    pass
                elif use_mmr:
                    embs: list[np.ndarray] = []
                    valid_rows: list[tuple] = []
                    for row in rows:
                        entity, entity_type, confidence = row[0], row[1], row[2]
                        emb_raw = row[3] if len(row) > 3 else None
                        try:
                            evec = _vec_to_numpy(emb_raw)
                        except (ValueError, TypeError, IndexError):
                            text = f"{entity} {entity_type}"
                            evec = np.asarray(
                                embedder.encode(str(text), convert_to_numpy=True),
                                dtype=np.float64,
                            ).ravel()
                        embs.append(evec)
                        valid_rows.append((entity, entity_type, confidence))
                    if embs:
                        doc_mat = np.stack(embs, axis=0)
                        order = _mmr_order(q_arr, doc_mat, limit, mmr_lambda)
                        for _, idx in enumerate(order, 1):
                            entity, entity_type, confidence = valid_rows[idx]
                            etype = str(entity_type).upper() if entity_type else "UNKNOWN"
                            content = f"Entity: {entity} ({entity_type}) with confidence {confidence}"
                            context_bits.append(content)
                            chunk_ids.append(str(entity))
                            metadata.append(
                                {"entity": str(entity) if entity else None, "entity_type": etype}
                            )
                else:
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
                sql = (
                    """
                    SELECT chunk_hash, content, article_id, embedding
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                    if use_mmr
                    else """
                    SELECT chunk_hash, content, article_id
                    FROM article_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                )
                cur.execute(sql, (vec_str, fetch_k))
                rows = cur.fetchall()
                if not rows:
                    pass
                elif use_mmr:
                    embs = []
                    valid_rows = []
                    for row in rows:
                        chunk_hash, content, _aid = row[0], row[1], row[2]
                        emb_raw = row[3] if len(row) > 3 else None
                        try:
                            evec = _vec_to_numpy(emb_raw)
                        except (ValueError, TypeError, IndexError):
                            evec = np.asarray(
                                embedder.encode(content or "", convert_to_numpy=True),
                                dtype=np.float64,
                            ).ravel()
                        embs.append(evec)
                        valid_rows.append((chunk_hash, content))
                    if embs:
                        doc_mat = np.stack(embs, axis=0)
                        order = _mmr_order(q_arr, doc_mat, limit, mmr_lambda)
                        for j, idx in enumerate(order, 1):
                            chunk_hash, content = valid_rows[idx]
                            chunk_ids.append(str(chunk_hash))
                            context_bits.append(f"[{j}] {content or ''}")
                            metadata.append({"entity": None, "entity_type": "TEXT"})
                else:
                    for i, row in enumerate(rows, 1):
                        chunk_hash, content, _ = row[0], row[1], row[2]
                        chunk_ids.append(str(chunk_hash))
                        context_bits.append(f"[{i}] {content or ''}")
                        metadata.append({"entity": None, "entity_type": "TEXT"})
        conn.close()
    except (psycopg2.OperationalError, psycopg2.Error, OSError) as e:
        logger.warning("get_vector_context failed: %s", e)
        error_slice = retrieval_slice_from_exception(
            e, backend="postgres", operation="get_vector_context"
        )
    except Exception as e:
        logger.warning("get_vector_context unexpected failure: %s", e)
        err = classify_exception(e, component="postgres", operation="get_vector_context")
        error_slice = RetrievalSlice(items=[], error=err)

    if use_mmr and context_bits:
        logger.info(
            "MMR: lambda=%.3f fetch_k=%d -> %d chunks (limit=%d)",
            mmr_lambda,
            fetch_k,
            len(context_bits),
            limit,
        )

    if error_slice is None and not context_bits:
        error_slice = RetrievalSlice(items=[], empty=True)

    return context_bits, chunk_ids, metadata, error_slice


def _get_default_retriever() -> UnifiedRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = UnifiedRetriever()
    return _default_retriever
