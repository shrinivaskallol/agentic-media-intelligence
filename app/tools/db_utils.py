"""
Database connection utilities for Postgres (agent_sql) and Neo4j (agent_graph).
Uses app.config.settings for validated configuration.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import psycopg2

from app.config.settings import get_settings
from app.errors.helpers import classify_exception
from app.errors.models import ConnectionSlice


def get_pg_conn():
    """Return a psycopg2 connection. Caller must close when done."""
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


def connect_postgres_result() -> ConnectionSlice:
    """Structured Postgres connection result (preferred over connect_postgres)."""
    try:
        return ConnectionSlice(resource="postgres", connection=get_pg_conn())
    except (psycopg2.OperationalError, psycopg2.Error, OSError) as e:
        return ConnectionSlice(
            resource="postgres",
            error=classify_exception(e, component="postgres", operation="connect"),
        )


def connect_postgres():
    """
    Return a psycopg2 connection, or None on failure.

    .. deprecated::
        Use :func:`connect_postgres_result` for structured ``ToolError`` semantics.
    """
    warnings.warn(
        "connect_postgres() is deprecated; use connect_postgres_result() for structured errors.",
        DeprecationWarning,
        stacklevel=2,
    )
    return connect_postgres_result().connection


@contextmanager
def get_postgres_connection():
    """Yield a psycopg2 connection. Caller must not close; context manager handles it."""
    slice_ = connect_postgres_result()
    if not slice_.ok:
        msg = slice_.error.message if slice_.error else "Failed to connect to Postgres"
        raise ConnectionError(msg)
    try:
        yield slice_.connection
    finally:
        slice_.connection.close()
