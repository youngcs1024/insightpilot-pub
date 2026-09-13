"""Global source registry and persisted active corpus pointer."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """Metadata is validated JSON text; source rows are never user-owned."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'deleted')", name="status"),
        CheckConstraint("chunk_count >= 0", name="chunk_count"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_path: Mapped[str] = mapped_column(String(1024), unique=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64))
    content_sha256: Mapped[str] = mapped_column(String(64))
    document_version: Mapped[str] = mapped_column(String(64))
    chunking_version: Mapped[str] = mapped_column(String(64))
    business_metadata: Mapped[str] = mapped_column(Text)
    chunk_count: Mapped[int]
    status: Mapped[str] = mapped_column(String(16))
    cleanup_pending: Mapped[bool]
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CorpusManifestRecord(Base, TimestampMixin):
    """A single atomic manifest rather than process-local active-version state."""

    __tablename__ = "corpus_manifests"
    __table_args__ = (CheckConstraint("id = 1", name="singleton"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    corpus_version: Mapped[str] = mapped_column(String(64))
    payload: Mapped[str] = mapped_column(Text)
