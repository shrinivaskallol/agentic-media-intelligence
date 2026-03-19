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

### Workflow Logic

| Node | Responsibility | Principal Decision |
|------|----------------|---------------------|
| Extractor | Identifies entities and intent | Short-circuits to END if intent is IRRELEVANT |
| Retrieve | Hybrid context retrieval | Fetches graph + vector evidence |
| Grader | Pre-synthesis quality check | Routes to Rewrite if context insufficient |
| Rewrite | Query refinement | Produces sharper query for re-retrieval |
| Synthesis | Answer generation | Routes to Evaluator or Critique on refusal |
| Evaluator | RAGAS faithfulness check | Routes to Critique if score < 0.8 |
| Critique | Error correction | Forces regeneration with specific feedback |
| Fallback | Safety net | Graceful response after max revisions |

### Trace Example: The Multi-Hop Challenge

**Query:** "Identify the primary optics provider for Apple's 3nm chip manufacturer."

| Step | Action |
|------|--------|
| **Extraction** | Maps "3nm manufacturer" → TSMC. Resolves intent to a Graph query: `MATCH (c:Company {name: 'TSMC'})-[:SUPPLIES]->(a:Company {name: 'Apple'})` |
| **Initial Retrieval** | Confirms TSMC as manufacturer in Neo4j but finds no direct "optics" link. |
| **Grader** | Flags "Context Insufficient" for the optics requirement; routes to Rewrite. |
| **Rewrite** | Pivots: "Who supplies lithography optics to ASML for TSMC's 3nm process?" |
| **Re-Retrieval** | Traverses 3-hop: Apple ← TSMC ← ASML ← Carl Zeiss AG. |
| **Synthesis** | Delivers grounded, multi-tier intelligence report. |

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
uv run python scripts/init_db.py
```

Seeds the Knowledge Graph (Postgres, Neo4j, Redis).

### 4. Test

```bash
uv run python scripts/test_agentic_loop.py
```

Observes the self-correction loop in real time.

---

## Project Structure

```
├── compose.yaml          # Neo4j, Postgres, Redis
├── pyproject.toml        # Dependencies (uv)
├── app/
│   ├── state/            # GraphState schema
│   ├── graph/            # LangGraph workflow definition
│   ├── nodes/            # Extractor, Retriever, Grader, Synthesis, Evaluator, Critique
│   └── tools/            # Vector + Graph retrieval
└── scripts/
    ├── init_db.py        # Seed databases
    └── test_agentic_loop.py
```

---

## Services (after `docker compose up -d`)

| Service | Port |
|---------|------|
| Neo4j Browser | http://localhost:7474 |
| Postgres | localhost:5432 |
| Redis | localhost:6379 |

Credentials are configured in `compose.yaml` and `.env`.

**MCP / Claude Desktop:** If the auto-intel tool disappears after an update, run `./scripts/fix_mcp_claude.sh` then restart Claude. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
