# Agentic Media Intelligence

AMI is an evidence-backed financial claim intelligence system that tracks what media sources said, when they said it, what evidence supported it, and whether dated predictions later resolved correctly.

> This branch contains the AMI V2 foundation. The original Corrective RAG / GraphRAG implementation remains preserved in the repository history and legacy application directories while V2 is developed incrementally under `v2/`.

## Why AMI

Financial commentary is fragmented across YouTube, news, filings, earnings calls, and market data. AMI converts those sources into an auditable claim ledger with provenance, evidence, outcomes, and source-level accuracy history.

AMI separates four concepts deliberately:

```text
observed facts -> interpretations/claims -> predictions -> realized outcomes
```

Observed market movement is recorded as an outcome, not automatically treated as proof of causality.

## Current target

The first vertical slice uses **Applied Digital Corporation (`APLD`)**.

```text
Finnhub + SEC EDGAR + YouTube + earnings/IR + price data
                         |
                         v
                    RawDocument
                         |
                         v
                claim/prediction extraction
                         |
                         v
               evidence verification
                         |
                         v
                prediction resolution
                         |
                         v
          cited digest + source scorecard
```

## Engineering pillars

1. **Data pipeline and platform**: incremental ingestion, idempotency, lineage, validation, retries, backfills, failed-record recovery.
2. **LLM and agent system**: typed LangGraph state, structured outputs, provenance, retrieval, verification, bounded loops, HITL.
3. **Production reliability and deployment**: evaluation gates, tracing, latency/cost telemetry, fallbacks, CI, APIs, deployment.

## Repository structure

```text
v2/
├── AGENTS.md                 # coding-agent engineering contract
├── README.md                 # V2 product and engineering overview
├── pyproject.toml            # standalone V2 package/test config
├── docs/
│   ├── architecture.md
│   ├── domain-model.md
│   └── roadmap.md
├── src/ami/
│   ├── domain/               # Pydantic domain contracts
│   ├── ingestion/            # source adapters + normalization
│   ├── processing/           # cleaning, enrichment, claim extraction
│   ├── retrieval/            # evidence retrieval/ranking
│   ├── workflows/            # bounded LangGraph workflows
│   ├── evaluation/           # gold set + regression evaluation
│   ├── storage/              # persistence + lineage
│   └── api/                  # transport layer
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evals/
└── data/                     # local fixtures/gold data only
```

## Development principles

- preserve raw source payloads before transformation
- every factual claim must retain provenance
- deterministic tasks stay deterministic
- LLM outputs validate against typed schemas
- agent loops are bounded
- retrieval and extraction are evaluated independently
- important prompt/model/schema changes are regression-gated
- infrastructure is added only when a measured requirement justifies it

## Roadmap

1. APLD raw ingestion and normalization
2. claim/prediction extraction + gold set
3. evidence retrieval and verification
4. prediction lifecycle and source scorecards
5. bounded LangGraph orchestration
6. production hardening and deployment

See [`v2/docs/roadmap.md`](v2/docs/roadmap.md) for milestone exit criteria and [`v2/AGENTS.md`](v2/AGENTS.md) for coding-agent rules.

## Non-goals

AMI V2 does not autonomously trade, generate unsupported price targets, or claim causal attribution from simple market correlations.
