# AMI V2: Agentic Media Intelligence

AMI V2 is a production-oriented system for tracking dated financial claims, linking them to evidence, resolving predictions against later outcomes, and maintaining an auditable source-accuracy history.

## Product boundary

AMI performs evidence synthesis and verification. It does not generate autonomous trades or ungrounded investment recommendations.

## Core flow

```text
raw sources
  -> normalize
  -> extract entities and claims
  -> validate structured records
  -> persist provenance
  -> retrieve supporting/contradicting evidence
  -> verify claim or prediction
  -> resolve against later outcomes
  -> update source scorecards
  -> produce cited digest
```

## V1 scope

- 1 initial company: Applied Digital Corporation (`APLD`)
- YouTube creator transcripts
- SEC EDGAR filings
- company/news data from Finnhub free-tier endpoints where available
- earnings releases/calls
- historical daily price data
- hand-labeled gold set for claim extraction and resolution

## Three engineering pillars

### 1. Data pipeline and platform

Incremental ingestion, source-normalized raw documents, idempotent processing, lineage, deduplication, schema validation, retries, backfills, failed-record recovery, freshness tracking.

### 2. LLM and agent system

LangGraph state machine, Pydantic structured outputs, claim extraction, retrieval, evidence verification, citation mapping, bounded transitions, confidence thresholds, HITL for ambiguity.

### 3. Production reliability and deployment

Evaluation gates, tracing, latency/cost telemetry, retries, fallbacks, timeouts, caching, CI, FastAPI, dashboard, reproducible environments.

## Repository layout

```text
v2/
├── README.md
├── AGENTS.md
├── docs/
│   ├── architecture.md
│   ├── domain-model.md
│   └── roadmap.md
├── src/ami/
│   ├── domain/
│   ├── ingestion/
│   ├── processing/
│   ├── retrieval/
│   ├── workflows/
│   ├── evaluation/
│   ├── storage/
│   └── api/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evals/
└── data/
    ├── raw/.gitkeep
    ├── fixtures/.gitkeep
    └── gold/.gitkeep
```

## Design rules

1. Preserve the original source payload before transformation.
2. Every derived claim must retain provenance to the source document and source span.
3. Deterministic tasks stay deterministic.
4. LLM outputs must validate against Pydantic models.
5. No unbounded agent loops.
6. Retrieval and claim extraction are evaluated independently.
7. Important prompt/model/schema changes must pass regression tests before merge.

## First vertical slice

For `APLD`:

1. ingest raw Finnhub, EDGAR, YouTube, earnings, and price data
2. normalize into a common `RawDocument` contract
3. extract claims and predictions
4. manually label a small gold set
5. verify/resolution workflow
6. generate cited company digest and creator scorecard

## Status

Foundation branch only. Legacy implementation remains untouched while V2 contracts and architecture are established.
