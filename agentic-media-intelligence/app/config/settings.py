"""
Application settings via Pydantic Settings.
Loads from .env with validation. Single source of truth for DB, Redis, and API keys.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Project root (directory containing pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Loaded from .env with validation."""

    model_config = SettingsConfigDict(
        env_file=_project_root() / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres
    postgres_user: str = Field(default="admin", alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="market_intel", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: str = Field(default="5433", alias="POSTGRES_PORT")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str | None = Field(default=None, alias="NEO4J_PASSWORD")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # LLM API keys (optional for tests)
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    @property
    def postgres_connection_params(self) -> dict:
        """Params for psycopg2.connect()."""
        if self.database_url:
            return {"dsn": self.database_url}
        return {
            "dbname": self.postgres_db,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "host": self.postgres_host,
            "port": self.postgres_port,
        }

    def get_database_url(self) -> str | None:
        """Return DATABASE_URL if set, else build from POSTGRES_* vars."""
        if self.database_url:
            return self.database_url
        if self.postgres_password:
            return (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return None


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
