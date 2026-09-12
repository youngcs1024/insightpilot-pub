"""Establish an independent history; bootstrap owns biz, Step 2.1 owns its data tables."""

revision: str = "0000_business_baseline"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Record the baseline without changing bootstrap-owned schemas or grants."""


def downgrade() -> None:
    """Remove the baseline revision without deleting bootstrap-owned objects."""
