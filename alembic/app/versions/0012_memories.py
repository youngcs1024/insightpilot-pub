"""Typed memories with mandatory provenance and restricted history updates."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0012_memories"
down_revision: str | None = "0011_evidence_audit"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create durable user preferences without granting content updates or deletion."""
    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "memory_type",
            sa.Enum(
                "metric_override",
                "region_focus",
                "terminology",
                "format_preference",
                name="memory_type",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("content", JSONB(), nullable=False),
        sa.Column("summary", sa.String(200), nullable=False),
        sa.Column("source_turn_id", sa.Uuid(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("superseded_by", sa.Uuid(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_turn_id"], ["turns.id"]),
        sa.ForeignKeyConstraint(["superseded_by"], ["memories.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        sa.CheckConstraint(
            "(is_active AND superseded_by IS NULL AND superseded_at IS NULL) OR "
            "(NOT is_active AND superseded_by IS NOT NULL AND superseded_at IS NOT NULL)",
            name="supersession_state",
        ),
        sa.CheckConstraint(
            "superseded_by IS NULL OR superseded_by <> id", name="no_self_supersede"
        ),
    )
    op.create_index(
        "ix_memories_user_type_active", "memories", ["user_id", "memory_type", "is_active"]
    )
    op.execute("REVOKE UPDATE, DELETE ON public.memories FROM app_rw")
    op.execute(
        "GRANT UPDATE (is_active, superseded_by, superseded_at, updated_at) "
        "ON public.memories TO app_rw"
    )


def downgrade() -> None:
    """Remove only memory storage during an explicit application schema rollback."""
    op.drop_table("memories")
