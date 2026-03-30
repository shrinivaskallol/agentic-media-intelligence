# Contributing to Agentic Media Intelligence

Thanks for your interest in contributing. This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/shrinivaskallol/agentic-media-intelligence.git
   cd agentic-media-intelligence
   uv sync
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and DB credentials
   ```

3. **Start infrastructure**

   ```bash
   docker compose up -d
   ```

4. **Bootstrap databases**

   ```bash
   uv run ami-migrate       # Postgres schema (news_articles, article_chunks, pgvector)
   uv run ami-init-db       # Neo4j + Redis (Postgres tables already from migrate)
   uv run ami-seed-postgres # Synthetic vector data
   uv run ami-seed-neo4j    # Synthetic graph data
   ```

   Or use the script paths: `uv run python scripts/init_db.py` etc.

## Code Quality (PEP 8)

- **Lint**: `uv run ruff check app/ prompts/ tests/ scripts/`
- **Format**: `uv run ruff format app/ prompts/ tests/ scripts/`
- **Pre-commit**: Run `pre-commit install` — on each commit, Ruff auto-fixes lint issues and formats code to PEP 8

## Testing

- **Unit tests** (matches CI): `uv run pytest tests/ -v -m "not integration"`
- **Integration tests** (Docker + DBs + API keys): `uv run pytest tests/ -v -m integration`

## Security

Do **not** open issues or PRs that contain **API keys**, `.env` contents, or production credentials. Rotate any key that was exposed. For responsible disclosure of security issues, contact the maintainer via the email on their [GitHub profile](https://github.com/shrinivaskallol) or LinkedIn from the main README.

## Pull Request Process

1. Create a branch from `main`
2. Make your changes; ensure lint and tests pass
3. Open a PR with a clear description of the change
4. Address review feedback

## Project Structure

- `app/` — Core application (nodes, graph, tools, config)
- `prompts/` — Versioned prompt registry (YAML per node)
- `scripts/` — Operational scripts (seed, init, run workflow)
- `tests/` — Pytest tests (unit + integration)

## Maintainer checklist (before “Public” on GitHub)

- [ ] **CI green** on `main` / `master` (Ruff + unit tests).
- [ ] **No `.env` in git history** — use `git log --all --diff-filter=A --name-only -- .env` (exact path); empty output is clean. *Note:* `git log -- '.env*'` also lists **`.env.example`**, which is meant to be tracked.
- [ ] **Rotate keys** if `.env` was ever leaked (chat, screenshot, gist).
- [ ] **Hero image** — optional real `docs/streamlit-executive-summary.png` ([docs/DEMO.md](docs/DEMO.md)).
