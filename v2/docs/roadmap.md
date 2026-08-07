# AMI V2 Roadmap

## Milestone 1: raw data foundation

Goal: prove reliable, reproducible ingestion for one company (`APLD`).

Deliverables:

- `RawDocument` contract
- Finnhub free-tier ingestion
- SEC EDGAR ingestion
- YouTube transcript ingestion
- earnings/IR ingestion
- historical daily-price ingestion
- idempotent persistence
- retries/rate-limit handling
- fixture-based unit tests

Exit criteria:

- rerunning ingestion does not duplicate records
- raw payloads are reproducible and source-attributed
- failures are observable and recoverable

## Milestone 2: claim extraction

Goal: convert raw text into provenance-preserving structured claims.

Deliverables:

- Pydantic claim/prediction schemas
- LLM structured extraction
- deterministic validation
- source-span mapping
- 20 to 30 hand-labeled APLD examples
- extraction precision/recall baseline

## Milestone 3: evidence retrieval and verification

Goal: independently evaluate whether relevant support/contradiction can be retrieved.

Deliverables:

- pgvector retrieval baseline
- metadata filters
- reranking only if metrics justify it
- Recall@K, MRR/context precision
- citation correctness
- confidence-gated verifier

## Milestone 4: prediction lifecycle

Goal: resolve dated predictions against later observable facts.

Deliverables:

- explicit resolution rules
- unresolved/resolved state transitions
- scheduled rechecks
- outcome evidence
- correct/incorrect/partial/unclear labels
- creator/source scorecards

## Milestone 5: agent workflow

Goal: orchestrate semantic tasks with bounded LangGraph state transitions.

Deliverables:

- typed state
- extract -> retrieve -> verify -> escalate/synthesize flow
- bounded retries
- HITL for ambiguity
- checkpointing
- traceable citations

## Milestone 6: production hardening

Goal: demonstrate operational ownership.

Deliverables:

- FastAPI
- Docker Compose
- CI evaluation gates
- tracing
- latency/token/cost metrics
- cache strategy
- health checks
- load/smoke tests
- dashboard

## V2 rule

Do not start the next milestone until the current milestone has measurable exit criteria and tests.
