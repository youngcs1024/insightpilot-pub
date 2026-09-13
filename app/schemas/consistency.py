"""Versioned, text-free maintenance reports and internal consistency projections."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, computed_field

from app.schemas.ingestion import (
    ActiveManifest,
    ChunkIdentity,
    Digest,
    RegisteredChunk,
    RegisteredDocument,
)
from app.schemas.mcp import Contract


class DriftKind(StrEnum):
    """The four operator-facing drift classes."""

    MISSING = "missing_reference"
    ORPHAN = "orphan_vector"
    COUNT = "document_count"
    SHA = "content_sha256"


class DriftReason(StrEnum):
    """Machine-readable causes, never decisions based on diagnostic prose."""

    NULL_PK = "null_pk"
    ABSENT_PK = "absent_pk"
    IDENTITY = "identity_mismatch"
    ORPHAN = "unknown_chunk"
    COUNTS = "counts_differ"
    EXTRA = "unreferenced_vector"
    REGISTRY = "registry_version"
    SHA = "hashes_differ"
    TEXT = "text_hash_differs"


class Drift(Contract):
    """One located finding; category counts count findings, not distinct chunks."""

    kind: DriftKind
    reason: DriftReason
    document_id: UUID
    chunk_uuid: UUID | None = None
    milvus_pk: int | None = None
    declared_count: int | None = None
    registry_count: int | None = None
    vector_count: int | None = None


class StoredChunk(ChunkIdentity):
    """Raw searchable text is hashed inside the adapter and never returned."""

    milvus_pk: int = Field(ge=0)
    actual_sha256: Digest


class IndexSnapshot(Contract):
    """An absent collection is distinct from a failed or incomplete scan."""

    exists: bool
    rows: list[StoredChunk] = Field(default_factory=list)


class RegistrySnapshot(Contract):
    """Detached state read while the corpus guard prevents concurrent publication."""

    documents: list[RegisteredDocument]
    chunks: list[RegisteredChunk]
    manifest: ActiveManifest | None


class Assessment(Contract):
    """Internal repair selection, derived from a complete validated scan."""

    drift: list[Drift] = Field(default_factory=list)
    rebuild: set[UUID] = Field(default_factory=set)


class RepairBlock(Contract):
    """Safe per-document failure codes; source text and exception prose are excluded."""

    document_id: UUID
    code: str


class ConsistencyReport(Contract):
    """Before/after findings make partial repair and final success unambiguous."""

    schema_version: Literal[1] = 1
    corpus_version: Digest | None = None
    collection_exists: bool = False
    fix_requested: bool = False
    before: list[Drift] = Field(default_factory=list)
    remaining: list[Drift] = Field(default_factory=list)
    blocked: list[RepairBlock] = Field(default_factory=list)
    cleanup_pending: list[UUID] = Field(default_factory=list)
    chunks_inserted: int = Field(default=0, ge=0)
    chunks_deleted: int = Field(default=0, ge=0)
    documents_repaired: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic serializes this property.
    @property
    def counts(self) -> dict[DriftKind, int]:
        """Always include every drift category, including zero counts."""
        return {kind: sum(item.kind is kind for item in self.remaining) for kind in DriftKind}

    @computed_field  # type: ignore[prop-decorator]  # Pydantic serializes this property.
    @property
    def successful(self) -> bool:
        """Only a complete clean rescan with no unresolved work passes."""
        return not (self.remaining or self.blocked or self.cleanup_pending)
