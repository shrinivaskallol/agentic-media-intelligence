"""
Database connection utilities for Postgres (agent_sql) and Neo4j (agent_graph).
Uses app.config.settings for validated configuration.
"""

from contextlib import contextmanager

import psycopg2

from app.config.settings import get_settings


def get_pg_conn():
    """Return a psycopg2 connection. Caller must close when done."""
    import psycopg2

    s = get_settings()
    params = s.postgres_connection_params
    if "dsn" in params:
        return psycopg2.connect(params["dsn"])
    return psycopg2.connect(**params)


def get_neo4j_driver():
    """Return a Neo4j driver instance. Caller should close when done."""
    from neo4j import GraphDatabase

    s = get_settings()
    return GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_username, s.neo4j_password),
    )


def connect_postgres():
    """Return a psycopg2 connection, or None on failure."""
    try:
        return get_pg_conn()
    except (psycopg2.OperationalError, psycopg2.Error, OSError):
        return None


@contextmanager
def get_postgres_connection():
    """Yield a psycopg2 connection. Caller must not close; context manager handles it."""
    conn = connect_postgres()
    if conn is None:
        raise ConnectionError("Failed to connect to Postgres")
    try:
        yield conn
    finally:
        conn.close()
