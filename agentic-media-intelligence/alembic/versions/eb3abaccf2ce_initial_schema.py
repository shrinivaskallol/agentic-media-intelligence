"""initial_schema

Revision ID: eb3abaccf2ce
Revises:
Create Date: 2026-03-19 19:21:42.301529

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "eb3abaccf2ce"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: news_articles, article_chunks (pgvector), indexes."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            title VARCHAR(512) NOT NULL,
            source VARCHAR(256),
            content TEXT,
            published_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS article_chunks (
            article_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),
            chunk_hash TEXT GENERATED ALWAYS AS (md5(article_id || '::' || content)) STORED,
            UNIQUE (chunk_hash)
        )
    """)

    op.execute("""
        ALTER TABLE article_chunks
        ADD COLUMN IF NOT EXISTS fts_tokens tsvector;
    """)
    op.execute("""
        UPDATE article_chunks SET fts_tokens = to_tsvector('english', content)
        WHERE fts_tokens IS NULL;
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_fts ON article_chunks USING GIN (fts_tokens);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_hnsw_embedding ON article_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_hnsw_embedding ON article_chunks;")
    op.execute("DROP INDEX IF EXISTS idx_fts ON article_chunks;")
    op.execute("DROP TABLE IF EXISTS article_chunks;")
    op.execute("DROP TABLE IF EXISTS news_articles;")
