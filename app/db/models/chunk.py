"""Deterministic active chunk registry, independent from answer snapshots."""

from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Chunk(Base, TimestampMixin):
    """Physical Milvus keys may change without changing logical chunk identities."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "document_version", "chunking_version", "ordinal"),
        Index("ix_chunks_sha", "content_sha256"),
        CheckConstraint("ordinal >= 0 AND char_len > 0", name="position_length"),
        CheckConstraint("page IS NULL OR page > 0", name="page"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document_version: Mapped[str] = mapped_column(String(64))
    chunking_version: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int]
    heading_path: Mapped[str] = mapped_column(String(512))
    page: Mapped[int | None]
    content_sha256: Mapped[str] = mapped_column(String(64))
    char_len: Mapped[int]
    milvus_pk: Mapped[int | None] = mapped_column(BigInteger, index=True)
