"""Persist typed clarification independently of evidence-backed answers."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0008_turn_clarification"
down_revision = "0007_metric_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Existing turns retain NULL clarification and their original answer."""
    op.add_column("turns", sa.Column("clarification", JSONB(none_as_null=True), nullable=True))


def downgrade() -> None:
    """Remove only the field owned by this revision."""
    op.drop_column("turns", "clarification")
