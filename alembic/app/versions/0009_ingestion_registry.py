"""Persist document/chunk identities and the atomic active corpus manifest."""

from datetime import datetime

import sqlalchemy as sa

from alembic import op

revision = "0009_ingestion_registry"
down_revision = "0008_turn_clarification"
branch_labels = None
depends_on = None


def timestamps() -> list[sa.Column[datetime]]:
    """Use the same server-owned timestamps as the application metadata."""
    return [
        sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
        for name in ("created_at", "updated_at")
    ]


def upgrade() -> None:
    """Add the complete registry without requiring a subsequent phase migration."""
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("source_path", sa.String(1024), nullable=False, unique=True),
        sa.Column("source_fingerprint", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("document_version", sa.String(64), nullable=False),
        sa.Column("chunking_version", sa.String(64), nullable=False),
        sa.Column("business_metadata", sa.Text, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("cleanup_pending", sa.Boolean, nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        *timestamps(),
        sa.CheckConstraint("status IN ('active', 'deleted')", name="status"),
        sa.CheckConstraint("chunk_count >= 0", name="chunk_count"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("document_id", sa.Uuid, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_version", sa.String(64), nullable=False),
        sa.Column("chunking_version", sa.String(64), nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading_path", sa.String(512), nullable=False),
        sa.Column("page", sa.Integer, nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("char_len", sa.Integer, nullable=False),
        sa.Column("milvus_pk", sa.BigInteger, nullable=True),
        *timestamps(),
        sa.UniqueConstraint("document_id", "document_version", "chunking_version", "ordinal"),
        sa.CheckConstraint("ordinal >= 0 AND char_len > 0", name="position_length"),
        sa.CheckConstraint("page IS NULL OR page > 0", name="page"),
    )
    op.create_index("ix_chunks_sha", "chunks", ["content_sha256"])
    op.create_index("ix_chunks_milvus_pk", "chunks", ["milvus_pk"])
    op.create_table(
        "corpus_manifests",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("corpus_version", sa.String(64), nullable=False),
        sa.Column("payload", sa.Text, nullable=False),
        *timestamps(),
        sa.CheckConstraint("id = 1", name="singleton"),
    )


def downgrade() -> None:
    """Remove only this revision's registry; historical answer evidence is untouched."""
    op.drop_table("corpus_manifests")
    op.drop_table("chunks")
    op.drop_table("documents")
