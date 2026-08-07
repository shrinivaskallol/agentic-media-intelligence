from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILED = "retryable_failed"
    PERMANENT_FAILED = "permanent_failed"


class SourceType(StrEnum):
    YOUTUBE = "youtube"
    SEC_EDGAR = "sec_edgar"
    FINNHUB = "finnhub"
    COMPANY_IR = "company_ir"
    EARNINGS_CALL = "earnings_call"
    MARKET_DATA = "market_data"
    NEWS = "news"


class RawDocument(BaseModel):
    document_id: str
    company_id: str
    source_id: str
    source_type: SourceType
    external_document_id: str
    title: str | None = None
    published_at: datetime | None = None
    ingested_at: datetime
    canonical_url: HttpUrl | None = None
    raw_text: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    content_checksum: str
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_version: str = "v1"


class Claim(BaseModel):
    claim_id: str
    document_id: str
    source_id: str
    company_id: str
    claim_text: str
    claim_type: str
    asserted_at: datetime | None = None
    source_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = "candidate"
