# AGENTS.md

This file defines the engineering contract for coding agents working on AMI V2.

## Objective

Build the smallest credible production-grade system that demonstrates:

1. data pipeline and platform ownership
2. production LLM/agent engineering
3. reliability, evaluation, and deployment

Do not optimize for feature count or resume-keyword density.

## Development rules

- Work in small, reviewable changes.
- Prefer explicit interfaces, schemas, state transitions, metrics, and failure handling.
- Do not introduce infrastructure without a concrete requirement.
- Preserve backward compatibility unless the task explicitly removes it.
- Do not write to secrets or commit credentials.
- Never fabricate production metrics or evaluation results.

## Agent design

- Use LangGraph for stateful reasoning workflows.
- Keep graph state explicit and typed.
- Bound all retry/rewrite/reflection loops.
- Use deterministic code for deterministic tasks.
- Use LLMs only for semantic extraction, interpretation, synthesis, or verification where deterministic logic is insufficient.
- Require Pydantic structured outputs for LLM-generated records.
- Preserve claim provenance through every state transition.
- Require citations for factual synthesis.
- Route low-confidence or ambiguous cases to HITL rather than forcing a decision.

## Data engineering

- Ingestion must be incremental, reproducible, and idempotent.
- Store raw source payloads before normalization.
- Track source identity, source URL/document ID, ingestion timestamp, source publication timestamp, checksum, processing version, and status.
- Support deduplication, retries, backfills, dead-letter/failed-record recovery, and freshness checks.
- Schema changes require migrations and regression coverage.

## Retrieval

- Treat retrieval as an independently evaluated subsystem.
- Start simple; add hybrid retrieval only when metrics justify it.
- Track Recall@K, MRR/context precision, and citation correctness.
- Prefer context selection and token-budget discipline over larger prompts.

## Evaluation

Maintain a hand-labeled gold set and evaluate at minimum:

- claim extraction precision/recall
- retrieval Recall@K and MRR/context precision
- faithfulness
- citation accuracy
- unsupported-claim rate
- task success
- prediction-resolution correctness

Changes to prompts, models, schemas, extraction logic, or retrieval logic must not silently regress the gold-set suite.

## Reliability

Implement appropriate:

- timeouts
- retries with backoff
- rate-limit handling
- idempotency
- validation
- fallbacks
- graceful degradation
- structured logging/tracing
- latency, token, and cost telemetry

## Testing expectations

For every feature:

1. define acceptance criteria
2. add or update tests
3. run targeted tests
4. run lint/type/schema checks when available
5. summarize changed behavior and known limitations

## Branch/commit discipline

- One coherent feature per branch or PR.
- Prefer conventional, specific commit messages such as `feat: ingest Finnhub company news`.
- Do not mix unrelated refactors with feature work.

## Current priority

The first V2 vertical slice is `APLD` raw-source ingestion and normalization. Do not build retrieval, Neo4j, multi-agent orchestration, or UI features ahead of the raw-data contract unless the task explicitly requires them.
