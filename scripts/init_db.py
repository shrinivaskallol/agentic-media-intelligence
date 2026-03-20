#!/usr/bin/env python3
"""Seed Pillar III: Postgres table, Neo4j nodes, Redis heartbeat — verifies infrastructure."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import psycopg2

from app import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def init_postgres() -> bool:
    """Create News Articles table in Postgres."""
    try:
        from app.config.settings import get_settings

        s = get_settings()
        url = s.get_database_url()
        if not url:
            logger.error("Postgres: Set DATABASE_URL or POSTGRES_* in .env")
            return False
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id SERIAL PRIMARY KEY,
                title VARCHAR(512) NOT NULL,
                source VARCHAR(256),
                content TEXT,
                published_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.close()
        conn.close()
        logger.info("Postgres: news_articles table ready")
        return True
    except (psycopg2.OperationalError, psycopg2.Error) as e:
        logger.error("Postgres: %s", e)
        return False


def init_neo4j() -> bool:
    """Create Company nodes in Neo4j."""
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable

        from app.config.settings import get_settings

        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session() as session:
            for name in ["Nvidia", "TSMC"]:
                session.run(
                    "MERGE (c:Company {name: $name}) SET c.updated_at = datetime()",
                    name=name,
                )
        driver.close()
        logger.info("Neo4j: Company nodes (Nvidia, TSMC) created")
        return True
    except (ServiceUnavailable, AuthError, OSError) as e:
        logger.error("Neo4j: %s", e)
        return False


def init_redis() -> bool:
    """Set a test heartbeat key in Redis."""
    try:
        import redis

        from app.config.settings import get_settings

        client = redis.from_url(get_settings().redis_url)
        client.set("heartbeat:pillar_iii", "ok", ex=3600)
        client.ping()
        logger.info("Redis: heartbeat key set")
        return True
    except (redis.ConnectionError, redis.RedisError, OSError) as e:
        logger.error("Redis: %s", e)
        return False


def main() -> None:
    """Run all seeds."""
    checks = [init_postgres(), init_neo4j(), init_redis()]
    if all(checks):
        logger.info("Pillar III operational. Agent can see.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
