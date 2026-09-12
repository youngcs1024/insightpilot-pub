"""Persist the exact versioned answer envelope for replay."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0005_turn_answer"
down_revision: str | None = "0004_data_evidence"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Existing turns remain readable without fabricated answer metadata."""
    op.add_column("turns", sa.Column("answer", postgresql.JSONB(none_as_null=True), nullable=True))


def downgrade() -> None:
    """Reverse only the new answer column."""
    op.drop_column("turns", "answer")
