# AMI V2 Architecture

## Architecture principle

AMI separates observed facts, interpretations, predictions, and realized outcomes. This prevents the system from conflating evidence with opinion or correlation with causation.

```text
SOURCE SYSTEMS
YouTube | EDGAR | Finnhub | Earnings/IR | Market Data
      |
      v
RAW INGESTION
immutable source payload + metadata + checksum
      |
      v
NORMALIZATION
RawDocument
      |
      v
SEMANTIC PROCESSING
entities + claims + predictions + evidence spans
      |
      v
PERSISTENCE
Postgres/pgvector; graph store only when graph queries justify it
      |
      +----------------------+
      |                      |
      v                      v
RETRIEVAL               RESOLUTION
support/contradict       later facts/outcomes
      |                      |
      +----------+-----------+
                 v
          LANGGRAPH WORKFLOW
      bounded verify/escalate/synthesize
                 |
                 v
             OUTPUTS
cited digest | claim ledger | source scorecard
```

## Boundary decisions

### Raw layer

The raw layer is append-oriented and source-faithful. It contains enough metadata to reproduce downstream processing.

### Normalized layer

Every source becomes a common `RawDocument` shape without destroying the original payload.

### Semantic layer

LLM extraction produces typed candidate claims and predictions. Validation and provenance are mandatory before persistence.

### Verification layer

Verification may retrieve supporting and contradicting evidence. A claim can remain `unclear`; the workflow must not force binary classification.

### Outcome layer

Prediction resolution compares the original dated prediction with later observable facts. Market movement is recorded as an outcome, not automatically treated as causal proof.

## V1 infrastructure

Use the smallest stack that proves the system:

- Python
- Pydantic
- Postgres + pgvector
- LangGraph
- FastAPI
- pytest
- Docker Compose
- structured tracing/telemetry

Neo4j is optional for V2 until graph traversal provides measurable retrieval value.

## Failure model

Each ingestion/processing record should support states such as:

```text
PENDING -> PROCESSING -> SUCCEEDED
                   |-> RETRYABLE_FAILED -> PROCESSING
                   |-> PERMANENT_FAILED
```

Failures must retain error type, attempt count, timestamps, and processing version.

## First implementation boundary

The first deployable slice stops after:

`APLD raw ingestion -> RawDocument normalization -> persistence -> tests`

Claim extraction is the next slice, not part of ingestion itself.
