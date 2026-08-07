# AMI V2 Domain Model

## Core entities

### Source

Represents an originator or publisher.

Examples: YouTube creator, SEC filing, company IR, news outlet, analyst.

Minimum fields:

- `source_id`
- `source_type`
- `name`
- `canonical_url`
- `external_id`

### RawDocument

Normalized envelope around an immutable raw payload.

Minimum fields:

- `document_id`
- `company_id`
- `source_id`
- `source_type`
- `external_document_id`
- `title`
- `published_at`
- `ingested_at`
- `canonical_url`
- `raw_text`
- `raw_payload`
- `content_checksum`
- `processing_status`
- `processing_version`

### Claim

A dated assertion attributable to a source.

Minimum fields:

- `claim_id`
- `document_id`
- `source_id`
- `company_id`
- `claim_text`
- `claim_type`
- `asserted_at`
- `source_span`
- `confidence`
- `status`

### Prediction

A claim that can be evaluated against future observable evidence.

Minimum fields:

- `prediction_id`
- `claim_id`
- `target_metric_or_event`
- `direction_or_expected_value`
- `horizon_start`
- `horizon_end`
- `resolution_rule`
- `resolution_status`

### Evidence

A source-backed fact or passage used to support, contradict, or contextualize a claim.

Minimum fields:

- `evidence_id`
- `document_id`
- `company_id`
- `evidence_text`
- `source_span`
- `relationship_to_claim`
- `observed_at`

### Outcome

An observable result used to resolve a prediction.

Minimum fields:

- `outcome_id`
- `prediction_id`
- `observed_value_or_event`
- `observed_at`
- `evidence_id`
- `resolution_label`
- `resolution_confidence`

## State vocabulary

Claims and predictions must support non-binary states.

Suggested claim status:

- `candidate`
- `validated`
- `rejected`
- `ambiguous`

Suggested prediction resolution status:

- `unresolved`
- `correct`
- `incorrect`
- `partially_correct`
- `unclear`
- `expired_unresolvable`

## Provenance invariant

Every persisted `Claim`, `Prediction`, `Evidence`, and `Outcome` must be traceable back to one or more source documents. A synthesized statement without provenance is not a valid factual AMI artifact.

## Causality invariant

Observed market movement is an outcome, not proof that a specific news item, filing, or claim caused that movement.
