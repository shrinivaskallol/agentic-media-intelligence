# agentic-media-intelligence

[![CI](https://github.com/shrinivaskallol/agentic-media-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/shrinivaskallol/agentic-media-intelligence/actions/workflows/ci.yml)

Stateful multi-agent system for financial news analysis using LangGraph, GraphRAG, and LLM evaluation. A live portfolio project by [Shri Kallol](https://www.linkedin.com/in/shrinivas-kallol/) demonstrating the transition from "Retrieval" to "Reasoning" in the 2026 AI Agent landscape.

Licensed under the [MIT License](LICENSE).

**Contents:** [Problem](#the-problem) · [Solution](#the-solution-agentic-design-patterns) · [Architecture](#system-architecture) · [RAGAS](#reliability-scorecard-ragas) · [Stack](#tech-stack) · [Getting started](#getting-started) · [MCP & HITL](#mcp-and-hitl) · [Streamlit](#streamlit-dashboard-optional) · [Layout](#project-structure) · [Ports](#services-after-docker-compose-up--d)

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

**Retrieval diversity (optional HITL)** (retrieve → diversity_gate → [human λ] → retrieve → grader)

- **Pattern:** Human-in-the-loop checkpoint + **Maximal Marginal Relevance (MMR)** on pgvector results
- **Function:** Vector retrieval can be re-ranked for diversity using λ ∈ [0.0, 1.0] (1.0 ≈ relevance-only; lower λ favors dissimilar chunks). When enabled via environment variables, the graph may **interrupt** after retrieval so an operator can submit λ; the workflow then **re-runs retrieval** with that λ before the Grader. Requires a LangGraph **checkpointer** (in-memory for the MCP server when HITL is on).

---

## System Architecture

![System Architecture](docs/architecture_graph.png)

*Figure 1: The Agentic Workflow — closed-loop evaluation and critique cycle (includes optional diversity gate / MMR re-fetch when HITL is enabled).*

**Updating this figure:** The README always points at `docs/architecture_graph.png`. Running the exporter **overwrites that file**; you do **not** need to edit the README image path.

```bash
uv run python scripts/export_graph.py
```

Writes **`docs/architecture_graph.mmd`** (plain Mermaid — verify `diversity_gate` here or at [mermaid.live](https://mermaid.live)) and **`docs/architecture_graph.png`** (needs **Graphviz** installed for PNG). Commit **both** files if you want the repo and GitHub README image to stay current.

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
| **Why MMR / optional HITL?** | Top-k similarity can return redundant passages. MMR trades off query relevance vs. diversity among selected chunks; optional interrupts let an operator set λ when redundancy risk is high (e.g. investigative review). |

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

### Clone → run (happy path)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/getting-started/installation/), Python **3.10–3.12** (managed by `uv`).

From an empty directory:

```bash
git clone https://github.com/shrinivaskallol/agentic-media-intelligence.git
cd agentic-media-intelligence
docker compose up -d
cp .env.example .env
```

Edit **`.env`**: set `POSTGRES_PASSWORD`, `NEO4J_PASSWORD` (must match what you use in Compose), at least one LLM key (`GOOGLE_API_KEY` and/or `GROQ_API_KEY`), and `DATABASE_URL` / Neo4j settings if your ports differ from the defaults.

```bash
uv sync
uv run ami-migrate        # Postgres (pgvector + article tables)
uv run ami-init-db        # Neo4j constraints + Redis ping
uv run ami-seed-postgres # Synthetic news chunks for embeddings
uv run ami-seed-neo4j    # Synthetic graph for GraphRAG
uv run pytest tests/ -v -m "not integration"   # optional: unit tests only
```

**Run the agent (pick one):**

| Goal | Command |
|------|---------|
| Terminal demo | `uv run ami-workflow` |
| MCP + SSE (tools / dashboard remote mode) | `uv run python src/mcp_server.py` |
| Streamlit UI | `uv run streamlit run app/ui/dashboard.py` (use **Remote** if MCP is running) |

**Integration tests** (Docker DBs + real keys): `uv run pytest tests/ -v -m integration`

### Portfolio demo (2–3 minutes)

Good for LinkedIn or interviews: show **Compose up → `.env` keys blurred → `uv sync` → seeds → one research query** in the Streamlit dashboard or MCP Inspector, then mention **CI** (lint + unit tests on GitHub Actions).

### 1. Infrastructure

```bash
docker compose up -d
```

Starts Neo4j, Postgres, and Redis.

### 2. Environment

Copy `.env.example` to `.env` and set passwords plus API keys. Never commit `.env`.

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

### 7. MCP server (development)

<a id="mcp-and-hitl"></a>

Run the MCP server with **SSE** on port 8000 (reachable on your LAN via `0.0.0.0`):

```bash
uv run python src/mcp_server.py
```

On startup, stderr prints `[MCP config] …` so you can confirm whether **HITL** is enabled (reads `.env`; see below).

- **Stdio** (Cursor “command” style): `uv run python src/mcp_server.py --stdio`
- **Port**: `MCP_PORT=9000` (default `8000`)

**MCP Inspector** (browser UI to list tools and call them):

```bash
npx @modelcontextprotocol/inspector http://localhost:8000/sse
```

Opens a UI (often at http://localhost:6274). Choose **SSE**, then **Connect**. Tools include **`research_company`** (query + `thread_id`) and **`resume_research`** (`thread_id` + `lambda_value`).

**Cursor:** Settings → Features → MCP → Add server → **SSE** → `http://localhost:8000/sse`.

**Observability**

| Mode | `research_company` | `resume_research` |
|------|---------------------|-------------------|
| HITL **off** | Streams node/tool events to the server terminal (`astream_events`) | N/A |
| HITL **on** | Runs until interrupt or completion (`ainvoke`); returns **JSON** with `"status": "interrupted"` when paused | Streams node/tool events to the server terminal (`astream_events`) |

**Human-in-the-loop (HITL) and MMR λ**

1. Copy **`.env.example` → `.env`** and set API keys. Never commit `.env`.
2. Enable diversity gate + interrupt path:

   | Variable | Purpose |
   |----------|---------|
   | `AMI_HITL_DIVERSITY` | `1`, `true`, or `yes` to enable; anything else disables HITL |
   | `AMI_HITL_MIN_CONTEXT` | Interrupt when **retrieval units** ≥ this value (integer ≥ 1): max of context blocks, vector chunk IDs, and `GRAPH FACT:` lines (default **3**) |

3. **Restart** `mcp_server.py` after changing `.env`. The server loads `.env` with override so these values take effect.

**Threshold note:** Older builds compared only `len(state.context)` (almost always 2), which made high defaults useless. Current behavior uses **retrieval units** (see table above); default **3** typically fires on a normal Nvidia/TSMC/Blackwell run with graph + vectors.

**Tool flow**

1. Call **`research_company`** with your **query** and a stable **`thread_id`** per conversation.
2. If the response is interrupt JSON → call **`resume_research`** with the **same `thread_id`** and **`lambda_value`** in **[0.0, 1.0]**.
3. `resume_research` completes the run and prints intermediate **node start** lines on the MCP process stderr.

**CLI smoke test (HITL + resume without Inspector):**

```bash
AMI_HITL_DIVERSITY=1 AMI_HITL_MIN_CONTEXT=2 uv run python scripts/test_hitl_mmr.py
```

**Dashboard SSE (same process as MCP):** optional HTTP routes for Streamlit or other UIs:

- `GET /ami/dashboard/stream?query=...&thread_id=...` — Server-Sent Events of node/tool markers + final state (or interrupt).
- `GET /ami/dashboard/resume/stream?thread_id=...&lambda_value=0.5` — continue after HITL.

Dependencies: `fastmcp`, `uvicorn` (see `pyproject.toml`).

### 8. Streamlit dashboard (optional)

```bash
# Terminal 1 — MCP must be up for Remote mode (SSE routes on MCP port, default :8000)
uv run python src/mcp_server.py

# Terminal 2
uv run streamlit run app/ui/dashboard.py
```

| Mode | What it does |
|------|----------------|
| **Remote (SSE → MCP)** | Live **pulse** (extract → retrieve → …) over `GET /ami/dashboard/stream` on the MCP server. |
| **Local (in-process)** | Runs `build_workflow()` in Streamlit; pulse after the run completes. |

**UI (portfolio-friendly):** Executive summary with **hoverable `[1]`, `[2]`** citations; **Sources & Evidence** grouped as **Semantic News (Vector)** vs **Structural facts (Graph)** (graph rows shown as plain English + *Verified via Knowledge Graph*). Engineers can open **Technical Trace (JSON)** (collapsed by default).

**HITL:** If `AMI_HITL_DIVERSITY=1`, the app may stop on **Diversity intervention**; pick **MMR λ** and **Resume with λ**. The sidebar **Diversity score** is `1 − λ` (how much retrieval biased toward diversity), not an automatic quality grade.

---

## Project Structure

```
├── alembic/              # Postgres migrations (news_articles, article_chunks)
├── compose.yaml          # Neo4j, Postgres, Redis
├── pyproject.toml        # Dependencies (uv)
├── docs/
│   ├── architecture_graph.mmd    # Generated Mermaid (source for the diagram)
│   └── architecture_graph.png   # Generated PNG for README (same export script)
├── src/
│   └── mcp_server.py     # MCP server (SSE on :8000 by default; --stdio for Cursor command)
├── app/
│   ├── state/            # GraphState schema (incl. mmr_lambda, HITL flags)
│   ├── graph/            # LangGraph workflow definition
│   ├── nodes/            # Extractor, Retriever, diversity_gate, Grader, Synthesis, …
│   ├── tools/            # Vector (MMR-capable) + Graph retrieval
│   └── ui/               # Streamlit dashboard + SSE event helpers for LangGraph
├── tests/                # Pytest tests (unit + integration)
│   ├── conftest.py       # Shared fixtures
│   ├── test_*.py
│   └── eval_dataset.json # Golden dataset for run_evaluation
└── scripts/              # Operational scripts (no tests)
    ├── init_db.py        # Bootstrap Postgres, Neo4j, Redis
    ├── seed_synthetic_postgres.py
    ├── seed_synthetic_neo4j.py
    ├── run_workflow.py   # Run CRAG workflow demo
    ├── run_evaluation.py # Run golden-dataset evaluation
    ├── export_graph.py   # Regenerate docs/architecture_graph.{mmd,png} from LangGraph
    └── test_hitl_mmr.py  # Optional local HITL + MMR smoke test
```

---

## Services (after `docker compose up -d`)

| Service | Port / URL |
|---------|------------|
| Neo4j Browser | http://localhost:7474 |
| Postgres (host → container) | **localhost:5433** → 5432 |
| Redis | localhost:6379 |

Match `DATABASE_URL` / `POSTGRES_PORT` in `.env` to **5433** on the host unless you change `compose.yaml`. Credentials are set in `compose.yaml` and `.env`.
