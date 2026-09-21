"""Immutable knowledge evidence with application ownership and idempotent turn identity."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0010_knowledge_evidence"
down_revision: str | None = "0009_ingestion_registry"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create snapshot storage and enforce append-only runtime access."""
    op.create_table(
        "knowledge_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("assistant_turn_id", sa.Uuid(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["assistant_turn_id"], ["turns.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "assistant_turn_id"),
    )
    op.execute("REVOKE UPDATE, DELETE ON public.knowledge_evidence FROM app_rw")


def downgrade() -> None:
    """Remove this application's snapshot table during explicit rollback."""
    op.drop_table("knowledge_evidence")
