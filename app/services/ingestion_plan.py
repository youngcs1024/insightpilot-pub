"""Read-only staging of manifest changes before model calls or persistent writes."""

from pathlib import Path

from pydantic import Field

from app.core.errors import CorpusValidationError
from app.schemas.ingestion import (
    DocumentStatus,
    FileFailure,
    PreparedDocument,
    RegisteredDocument,
    document_id,
)
from app.schemas.mcp import Contract
from app.services.chunking import prepare
from app.services.ingestion_config import IngestionSettings
from app.services.loaders import discover, load_source, snapshot


class IngestionPlan(Contract):
    """A failed present file retains its previous committed document and chunks."""

    changed: list[PreparedDocument] = Field(default_factory=list)
    refreshed: list[RegisteredDocument] = Field(default_factory=list)
    deleted: list[RegisteredDocument] = Field(default_factory=list)
    failed: list[FileFailure] = Field(default_factory=list)


def stage(
    root: Path, previous: list[RegisteredDocument], config: IngestionSettings
) -> IngestionPlan:
    """Fingerprint every listed resource; parse and split only changed snapshots."""
    inventory = discover(root)
    existing = {item.document_id: item for item in previous}
    present = {document_id(entry.path) for entry in inventory.documents}
    plan = IngestionPlan(
        deleted=[
            item.model_copy(
                update={"status": DocumentStatus.DELETED, "chunk_count": 0, "cleanup_pending": True}
            )
            for item in previous
            if item.document_id not in present and item.status is DocumentStatus.ACTIVE
        ]
    )
    for entry in inventory.documents:
        try:
            source = snapshot(root, entry, config.max_source_bytes)
            fingerprint = source.fingerprint
            old = existing.get(document_id(entry.path))
            if (
                old is not None
                and old.status is DocumentStatus.ACTIVE
                and old.source_fingerprint == fingerprint
                and old.chunking_version == config.chunking_version()
            ):
                continue
            prepared = prepare(load_source(entry, source.raw, source.sidecar, fingerprint), config)
            if (
                old is not None
                and old.status is DocumentStatus.ACTIVE
                and old.document_version == prepared.document.document_version
                and old.chunking_version == prepared.document.chunking_version
            ):
                plan.refreshed.append(old.model_copy(update={"source_fingerprint": fingerprint}))
            else:
                plan.changed.append(prepared)
        except CorpusValidationError as exc:
            plan.failed.append(FileFailure(path=entry.path, code=exc.code))
    return plan
