from datetime import UTC, datetime

from ami.domain.models import ProcessingStatus, RawDocument, SourceType


def test_raw_document_defaults_to_pending() -> None:
    document = RawDocument(
        document_id="doc_001",
        company_id="APLD",
        source_id="finnhub",
        source_type=SourceType.FINNHUB,
        external_document_id="example_001",
        ingested_at=datetime.now(UTC),
        content_checksum="abc123",
    )

    assert document.processing_status == ProcessingStatus.PENDING
    assert document.processing_version == "v1"
