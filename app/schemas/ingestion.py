"""Versioned ingestion boundaries and stable corpus identities."""

import hashlib
import json
import unicodedata
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.corpus import CorpusEntry, CorpusMetadata
from app.schemas.mcp import Contract
from app.schemas.model_runtime import Dense, ModelMetadata, Sparse

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def normalize(value: str) -> str:
    """Canonical text is NFC with LF line endings, without case folding."""
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def canonical(value: object) -> str:
    """Serialize explicit identity projections, never clocks or machine paths."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str) -> str:
    """Hash canonical UTF-8 content with SHA-256."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(path: str) -> UUID:
    """Paths are validated at discovery before entering this identity function."""
    return uuid5(NAMESPACE_URL, "insightpilot:corpus:" + normalize(path))


class SourcePart(Contract):
    """A Markdown body, Excel sheet or PDF page, retaining its original boundary."""

    text: str = Field(min_length=1)
    heading: str = ""
    page: int | None = Field(default=None, ge=1)


class SourceSnapshot(Contract):
    """Bound source bytes to their fingerprint without logging document contents."""

    raw: bytes = Field(repr=False)
    sidecar: bytes = Field(repr=False)
    fingerprint: Digest


class LoadedSource(Contract):
    """One source snapshot: fingerprint and parsed text derive from the same bytes."""

    entry: CorpusEntry
    metadata: CorpusMetadata
    source_fingerprint: Digest
    parts: list[SourcePart] = Field(min_length=1)


class VersionMember(Contract):
    """The exact fields in the shared corpus-version identity."""

    document_id: UUID
    document_version: Digest
    chunking_version: Digest


class EncodingProfile(Contract):
    """Only embedding semantics affect index compatibility; timing does not."""

    model: str
    revision: str
    precision: Literal["fp16", "fp32"]
    max_length: int = Field(ge=1, le=512)
    dense_normalized: Literal[True] = True

    @classmethod
    def from_metadata(cls, metadata: ModelMetadata) -> "EncodingProfile":
        """Project the effective server encoding configuration."""
        return cls(
            model=metadata.embed_model,
            revision=metadata.embed_revision,
            precision=metadata.precision,
            max_length=metadata.embed_max_length,
        )


class ActiveManifest(Contract):
    """One shared logical corpus per application database, independent of root path."""

    corpus_version: Digest
    members: list[VersionMember]
    splitter_configs: dict[Digest, str]
    encoding: EncodingProfile | None = None
    model_metadata: ModelMetadata | None = None
    collection: str

    @model_validator(mode="after")
    def verify_identity(self) -> Self:
        """A corrupted pointer must fail closed instead of admitting vector candidates."""
        members = [(str(item.document_id), item.document_version, item.chunking_version) for item in self.members]
        if members != sorted(members) or len({item.document_id for item in self.members}) != len(members):
            raise PydanticCustomError("manifest_members", "Invalid manifest membership")
        if digest(canonical(members)) != self.corpus_version:
            raise PydanticCustomError("manifest_digest", "Invalid manifest identity")
        if any(digest(value) != key for key, value in self.splitter_configs.items()):
            raise PydanticCustomError("manifest_splitter", "Invalid splitter identity")
        if any(item.chunking_version not in self.splitter_configs for item in self.members):
            raise PydanticCustomError("manifest_config", "Missing splitter configuration")
        return self


class DocumentStatus(StrEnum):
    """Removed documents retain a tombstone for crash-safe vector cleanup."""

    ACTIVE = "active"
    DELETED = "deleted"


class RegisteredDocument(VersionMember):
    """Detached registry projection; the source itself is global reference data."""

    source_path: str = Field(max_length=1024)
    source_fingerprint: Digest
    content_sha256: Digest
    metadata: CorpusMetadata
    chunk_count: int = Field(ge=0)
    status: DocumentStatus = DocumentStatus.ACTIVE
    cleanup_pending: bool = True


class ChunkIdentity(VersionMember):
    """All identity fields needed to validate a candidate, including its physical row."""

    chunk_uuid: UUID
    content_sha256: Digest
    milvus_pk: int | None = Field(default=None, ge=0)


class RegisteredChunk(ChunkIdentity):
    """PostgreSQL stores provenance, while Milvus owns searchable text."""

    ordinal: int = Field(ge=0)
    heading_path: str = Field(max_length=512)
    page: int | None = Field(default=None, ge=1)
    char_len: int = Field(gt=0)


class PreparedChunk(RegisteredChunk):
    """Fully validated child/parent payload before any model or storage write."""

    content: str = Field(min_length=1, max_length=32_000)
    parent_content: str = Field(min_length=1, max_length=65535)


class PreparedDocument(Contract):
    """One complete replacement; unchanged chunks of this file stay included."""

    document: RegisteredDocument
    chunks: list[PreparedChunk] = Field(min_length=1)


class VectorRow(Contract):
    """Typed Milvus insert input; BM25 is generated by the server."""

    chunk: PreparedChunk
    metadata: CorpusMetadata
    dense: Dense
    sparse_learned: Sparse


class InsertReceipt(Contract):
    """The complete ordered physical identity returned by a single insert."""

    ids: list[Annotated[int, Field(strict=True, ge=0)]]


class FileFailure(Contract):
    """Safe per-file diagnostic without backend prose or source content."""

    path: str
    code: str


class IngestionResult(Contract):
    """A committed partial batch is distinct from full operator success."""

    documents_changed: int = 0
    documents_deleted: int = 0
    chunks_inserted: int = 0
    chunks_deleted: int = 0
    failed_files: list[FileFailure] = Field(default_factory=list)
    cleanup_pending: list[UUID] = Field(default_factory=list)
    corpus_version: str | None = None

    @property
    def successful(self) -> bool:
        """A pending cleanup or failed file must produce a failing CLI exit."""
        return not self.failed_files and not self.cleanup_pending
