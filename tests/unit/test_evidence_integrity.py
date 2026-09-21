"""Digest validation must precede model defaults and safely reject corrupt records."""

from uuid import uuid4

import pytest

from app.agents.summarize import package_result
from app.core.errors import EvidenceIntegrityError
from app.db.models.evidence import DataEvidenceRecord, KnowledgeEvidenceRecord
from app.repositories.evidence import data_snapshot, knowledge_snapshot, payload_fingerprint
from app.retrieval.config import EvidenceConfig
from app.retrieval.evidence import package_evidence
from app.services.schema_tokens import SchemaTokenCounter
from tests.agents.knowledge_support import ranked
from tests.agents.support import result


@pytest.mark.parametrize("kind", ["data", "knowledge"])
@pytest.mark.parametrize("damage", ["digest", "version", "unknown", "payload", "boolean"])
def test_corrupt_evidence_is_typed(kind: str, damage: str) -> None:
    data = package_result(result(), [])
    knowledge = package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter())
    value = data if kind == "data" else knowledge
    record_type = DataEvidenceRecord if kind == "data" else KnowledgeEvidenceRecord
    record = record_type(
        id=uuid4(),
        schema_version=value.schema_version,
        payload=value.model_dump(mode="json"),
        content_sha256="",
    )
    if damage == "version":
        record.schema_version = 99
    elif damage == "unknown":
        record.schema_version = 99
        record.payload["schema_version"] = 99
    elif damage == "payload":
        record.payload.pop("sql" if kind == "data" else "chunks")
    elif damage == "boolean":
        record.payload["schema_version"] = True
        record.schema_version = 1
    record.content_sha256 = payload_fingerprint(record.payload)
    if damage == "digest":
        record.content_sha256 = "0" * 64
    decode = data_snapshot if kind == "data" else knowledge_snapshot
    with pytest.raises(EvidenceIntegrityError):
        decode(record)


def test_legacy_digest_precedes_added_model_defaults() -> None:
    evidence = package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter())
    payload = evidence.model_dump(mode="json")
    payload["schema_version"] = 1
    payload.pop("original_question")
    payload.pop("assumptions")
    record = KnowledgeEvidenceRecord(
        id=uuid4(), schema_version=1, payload=payload, content_sha256=payload_fingerprint(payload)
    )
    before = dict(payload)
    snapshot = knowledge_snapshot(record)
    assert snapshot.knowledge.schema_version == 1
    assert snapshot.knowledge.original_question is None
    assert record.payload == before
    assert record.content_sha256 == payload_fingerprint(before)
