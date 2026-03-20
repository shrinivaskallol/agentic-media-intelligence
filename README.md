# agentic-media-intelligence

Stateful multi-agent system for financial news analysis using LangGraph, GraphRAG, and LLM evaluation. A live portfolio project by [Shri Kallol](https://www.linkedin.com/in/shrinivas-kallol/) demonstrating the transition from "Retrieval" to "Reasoning" in the 2026 AI Agent landscape.

---

## The Problem

Standard RAG (Retrieval-Augmented Generation) is brittle. It suffers from:

- **Search Drift** — The first retrieval is irrelevant, and the system has no recourse.
- **Silent Hallucinations** — The model invents connections between entities not present in the evidence.

For complex financial and supply-chain queries, one-shot retrieval is a production risk.

---

## The Solution: Agentic Design Patterns

AMI is a Corrective RAG (CRAG) system built on a cyclic state machine. It does not just "search and tell"; it grades, rewrites, and reflects.

### Core Agentic Loops

**The Query Refinement Loop** (retrieve → grader → rewrite)

- **Pattern:** Corrective RAG (CRAG)
- **Function:** If the Grader node detects that retrieved context is insufficient or noisy, it triggers an autonomous Rewrite node to reformulate the search query based on missing entities.

**The Answer Quality Loop** (synthesis → evaluator → critique)

- **Pattern:** Self-Reflection
- **Function:** The Evaluator (using RAGAS metrics) acts as a unit test. If the synthesis is not fully grounded in the facts, the Critique node identifies the hallucination and forces a regeneration.

---

## System Architecture

![System Architecture](docs/architecture_graph.png)

*Figure 1: The Agentic Workflow — closed-loop evaluation and critique cycle.*

### Trace Example: The Multi-Hop Challenge

**Query:** "Identify the primary optics provider for Apple's 3nm chip manufacturer."

| Node | Responsibility | Action |
|------|----------------|--------|
| Extractor | Identifies entities and intent | Maps "3nm manufacturer" → TSMC. Resolves intent to a Graph query: `MATCH (c:Company {name: 'TSMC'})-[:SUPPLIES]->(a:Company {name: 'Apple'})` |
| Retrieve | Hybrid context retrieval | Confirms TSMC as manufacturer in Neo4j but finds no direct "optics" link. |
| Grader | Pre-synthesis quality check | Flags "Context Insufficient" for the optics requirement; routes to Rewrite. |
| Rewrite | Query refinement | Pivots: "Who supplies lithography optics to ASML for TSMC's 3nm process?" |
| Retrieve | Re-retrieval | Traverses 3-hop: Apple ← TSMC ← ASML ← Carl Zeiss AG. |
| Synthesis | Answer generation | Delivers grounded, multi-tier intelligence report. |

**Result:** The agent bridged a consumer-facing entity (Apple) and a deep-tier industrial supplier (Carl Zeiss AG) by inferring the implicit manufacturer (TSMC) and autonomously pivoting to the lithography layer (ASML) when the initial search hit a dead-end.

[View trace](https://smith.langchain.com/public/ba6c8ab5-a5a1-499b-8dd6-96482adbf2b3/r)

### Principal Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Why GraphRAG (Neo4j)?** | Vector search excels at nuance but fails at relationship traversal. To answer "Who is the optics provider for Apple's 3nm manufacturer?", the system must perform a 3-hop traversal. Neo4j provides the structural truth that vector DBs lack. |
| **Why Stateful LangGraph?** | A stateful managed graph maintains a Revision Count and Search History. This prevents infinite loops and allows the system to remember what it already tried during the rewrite phase. |
| **Why Hybrid Retrieval?** | Neo4j (structural) + PGVector (semantic) capture both hard links between companies and soft sentiment in news text. |

---

## Reliability Scorecard (RAGAS)

LLM-as-a-Judge is treated as a production requirement, not an afterthought. Metrics use P50 (typical case) and P95 (worst case) to reflect distribution.

| Metric | P50 (Typical) | P95 (Worst Case) | Principal Insight |
|--------|---------------|------------------|-------------------|
| Faithfulness | 0.94 | 0.81 | High grounding via Neo4j; drop-off on highly sparse news. |
| Answer Relevancy | 0.89 | 0.72 | Query rewriting ensures focus; lower on broad macro-economic queries. |
| Total Latency | ~7s | ~22s | Loop-driven; latency scales with the number of Rewrite cycles. |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Brain | Gemini 1.5 Flash (optimized for high-speed agentic reasoning) |
| Orchestration | LangGraph (Stateful Python SDK) |
| Databases | Neo4j (Knowledge Graph), Postgres (Vector Store), Redis (Persistence) |
| Tooling | uv for deterministic builds, Ragas for automated evaluation |

---

## Evidence of Resilience (Traces)

| Case | Description | Trace |
|------|-------------|-------|
| **Safe Refusal** | System recognizes missing data and refuses to hallucinate | [View trace](https://smith.langchain.com/public/5cf1d025-6669-4d13-863a-ab0d4d101041/r) |
| **Self-Correction Loop** | Query rewrite after a noisy first search | [View trace](https://smith.langchain.com/public/3ad62471-b85e-4c93-b499-f0f6559da417/r) |
| **GraphRAG Hop** | Successful 3-hop entity traversal through Neo4j | [View trace](https://smith.langchain.com/public/ba6c8ab5-a5a1-499b-8dd6-96482adbf2b3/r) |
| **MCP Tool (Claude Desktop)** | Auto-intel invoked as MCP tool from Claude Desktop | [View trace](https://claude.ai/share/c5793f76-386e-4ecc-bbba-65e11f9c99fb) |

---

## Getting Started

### 1. Infrastructure

```bash
docker compose up -d
```

Starts Neo4j, Postgres, and Redis.

### 2. Environment

Copy `.env.example` and add your API keys.

### 3. Bootstrap

```bash
uv sync
uv run ami-migrate       # Postgres schema (news_articles, article_chunks, pgvector)
uv run ami-init-db      # Neo4j + Redis
uv run ami-seed-postgres # Synthetic vector data
uv run ami-seed-neo4j   # Synthetic graph data
```

Or: `uv run python scripts/init_db.py` for init_db only.

### 4. Run

```bash
uv run ami-workflow
```

Observes the self-correction loop in real time. Alternatively: `uv run python scripts/run_workflow.py`

### 5. Test

```bash
uv run pytest tests/ -v
```

Integration tests (require Docker): `uv run pytest tests/ -v -m integration`

### 6. Lint & Pre-commit

```bash
uv run ruff check app/ prompts/ tests/ scripts/
pre-commit install   # optional: run hooks on git commit
```

---

## Project Structure

```
├── alembic/              # Postgres migrations (news_articles, article_chunks)
├── compose.yaml          # Neo4j, Postgres, Redis
├── pyproject.toml        # Dependencies (uv)
├── app/
│   ├── state/            # GraphState schema
│   ├── graph/            # LangGraph workflow definition
│   ├── nodes/            # Extractor, Retriever, Grader, Synthesis, Evaluator, Critique
│   └── tools/            # Vector + Graph retrieval
├── tests/                # Pytest tests (unit + integration)
│   ├── conftest.py       # Shared fixtures
│   ├── test_*.py
│   └── eval_dataset.json # Golden dataset for run_evaluation
└── scripts/              # Operational scripts (no tests)
    ├── init_db.py        # Bootstrap Postgres, Neo4j, Redis
    ├── seed_synthetic_postgres.py
    ├── seed_synthetic_neo4j.py
    ├── run_workflow.py   # Run CRAG workflow demo
    └── run_evaluation.py # Run golden-dataset evaluation
```

---

## Services (after `docker compose up -d`)

| Service | Port |
|---------|------|
| Neo4j Browser | http://localhost:7474 |
| Postgres | localhost:5432 |
| Redis | localhost:6379 |

Credentials are configured in `compose.yaml` and `.env`.
