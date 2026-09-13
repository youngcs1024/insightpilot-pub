"""Pure comparison of the authoritative manifest, registry and complete physical index."""

from collections import Counter

from app.core.errors import IngestionRegistryError
from app.schemas.consistency import (
    Assessment,
    Drift,
    DriftKind,
    DriftReason,
    IndexSnapshot,
    RegistrySnapshot,
    StoredChunk,
)
from app.schemas.ingestion import ChunkIdentity, DocumentStatus, EncodingProfile, RegisteredChunk


def validate_registry(registry: RegistrySnapshot, collection: str) -> None:
    """Never infer a new active manifest from damaged state or from vectors."""
    active = {
        (item.document_id, item.document_version, item.chunking_version)
        for item in registry.documents
        if item.status is DocumentStatus.ACTIVE
    }
    manifest = registry.manifest
    if manifest is None:
        if registry.documents or registry.chunks:
            raise IngestionRegistryError()
        return
    expected = {
        (item.document_id, item.document_version, item.chunking_version)
        for item in manifest.members
    }
    if active != expected or manifest.collection != collection:
        raise IngestionRegistryError()
    if active and (manifest.encoding is None or manifest.model_metadata is None):
        raise IngestionRegistryError()
    if (
        manifest.model_metadata is not None
        and EncodingProfile.from_metadata(manifest.model_metadata) != manifest.encoding
    ):
        raise IngestionRegistryError()


def same_identity(left: ChunkIdentity, right: ChunkIdentity) -> bool:
    """Hash differences have their own category, separate from linkage errors."""
    return (
        left.chunk_uuid == right.chunk_uuid
        and left.document_id == right.document_id
        and left.document_version == right.document_version
        and left.chunking_version == right.chunking_version
    )


def missing(chunk: RegisteredChunk, row: StoredChunk | None) -> Drift | None:
    """A populated but dangling or foreign physical key is still a missing link."""
    reason = None
    if chunk.milvus_pk is None:
        reason = DriftReason.NULL_PK
    elif row is None:
        reason = DriftReason.ABSENT_PK
    elif not same_identity(chunk, row):
        reason = DriftReason.IDENTITY
    if reason is None:
        return None
    return Drift(
        kind=DriftKind.MISSING,
        reason=reason,
        document_id=chunk.document_id,
        chunk_uuid=chunk.chunk_uuid,
        milvus_pk=chunk.milvus_pk,
    )


def inspect_row(row: StoredChunk, chunk: RegisteredChunk | None) -> list[Drift]:
    """Report corrupt hashes even when the damaged row is also an orphan or duplicate."""
    reasons: list[tuple[DriftKind, DriftReason]] = []
    if chunk is None:
        reasons.append((DriftKind.ORPHAN, DriftReason.ORPHAN))
    elif chunk.milvus_pk != row.milvus_pk or not same_identity(chunk, row):
        reasons.append((DriftKind.COUNT, DriftReason.EXTRA))
    if chunk is not None and chunk.content_sha256 != row.content_sha256:
        reasons.append((DriftKind.SHA, DriftReason.SHA))
    if row.actual_sha256 != row.content_sha256:
        reasons.append((DriftKind.SHA, DriftReason.TEXT))
    return [
        Drift(
            kind=kind,
            reason=reason,
            document_id=row.document_id,
            chunk_uuid=row.chunk_uuid,
            milvus_pk=row.milvus_pk,
        )
        for kind, reason in reasons
    ]


def assess(registry: RegistrySnapshot, index: IndexSnapshot) -> Assessment:
    """Find all drift and select documents requiring source-verified reconstruction."""
    result = Assessment()
    physical = {row.milvus_pk: row for row in index.rows}
    chunks = {chunk.chunk_uuid: chunk for chunk in registry.chunks}
    documents = {item.document_id: item for item in registry.documents}
    for chunk in registry.chunks:
        row = physical.get(chunk.milvus_pk) if chunk.milvus_pk is not None else None
        issue = missing(chunk, row)
        if issue is not None:
            result.drift.append(issue)
            result.rebuild.add(chunk.document_id)
        document = documents[chunk.document_id]
        if (
            document.status is DocumentStatus.DELETED
            or document.document_version != chunk.document_version
            or document.chunking_version != chunk.chunking_version
        ):
            result.drift.append(
                Drift(
                    kind=DriftKind.MISSING,
                    reason=DriftReason.REGISTRY,
                    document_id=chunk.document_id,
                    chunk_uuid=chunk.chunk_uuid,
                    milvus_pk=chunk.milvus_pk,
                )
            )
            result.rebuild.add(chunk.document_id)
    for row in index.rows:
        chunk = chunks.get(row.chunk_uuid)
        issues = inspect_row(row, chunk)
        result.drift.extend(issues)
        if (
            chunk is not None
            and chunk.milvus_pk == row.milvus_pk
            and any(issue.kind is DriftKind.SHA for issue in issues)
        ):
            result.rebuild.add(chunk.document_id)
    compare_counts(registry, index, result)
    return result


def compare_counts(
    registry: RegistrySnapshot, index: IndexSnapshot, result: Assessment
) -> None:
    """Extra vectors can be cleaned without encoding; missing registry rows cannot."""
    registered = Counter(chunk.document_id for chunk in registry.chunks)
    physical = Counter(row.document_id for row in index.rows)
    for document in registry.documents:
        identifier = document.document_id
        expected = document.chunk_count if document.status is DocumentStatus.ACTIVE else 0
        if not document.chunk_count == expected == registered[identifier] == physical[identifier]:
            result.drift.append(
                Drift(
                    kind=DriftKind.COUNT,
                    reason=DriftReason.COUNTS,
                    document_id=identifier,
                    declared_count=document.chunk_count,
                    registry_count=registered[identifier],
                    vector_count=physical[identifier],
                )
            )
        if document.chunk_count != expected or expected != registered[identifier]:
            result.rebuild.add(identifier)
