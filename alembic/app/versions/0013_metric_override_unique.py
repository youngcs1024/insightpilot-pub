"""Enforce one committed active metric override per user without rewriting history."""

import sqlalchemy as sa

from alembic import op

revision: str = "0013_metric_override_unique"
down_revision: str | None = "0012_memories"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Legacy duplicates abort this transaction; administrators resolve them explicitly."""
    op.add_column(
        "memories",
        sa.Column(
            "active_metric_key",
            sa.String(64),
            sa.Computed(
                "(CASE WHEN is_active AND memory_type::text = 'metric_override'::text "
                "THEN content ->> 'metric_key'::text ELSE NULL::text END)::character varying(64)",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_unique_constraint(
        "uq_memories_active_metric",
        "memories",
        ["user_id", "memory_type", "active_metric_key"],
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    """Retain every memory version while removing this constraint and derived column."""
    op.drop_constraint("uq_memories_active_metric", "memories", type_="unique")
    op.drop_column("memories", "active_metric_key")
