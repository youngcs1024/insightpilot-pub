"""Backfill explicit evidence versions without rewriting historical JSON or digests."""

import sqlalchemy as sa

from alembic import op

revision: str = "0011_evidence_audit"
down_revision: str | None = "0010_knowledge_evidence"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Retain append-only runtime grants while the owner backfills original versions."""
    for table in ("data_evidence", "knowledge_evidence"):
        op.add_column(table, sa.Column("schema_version", sa.Integer(), nullable=True))
        # Identifiers come only from this migration's fixed table names.
        op.execute(
            sa.text(
                "UPDATE public." + table
                + " SET schema_version = (payload ->> 'schema_version')::integer"
            )
        )
        op.alter_column(table, "schema_version", existing_type=sa.Integer(), nullable=False)
        op.execute(sa.text("REVOKE UPDATE, DELETE ON public." + table + " FROM app_rw"))


def downgrade() -> None:
    """Only remove redundant audit columns; preserve every original snapshot."""
    for table in ("knowledge_evidence", "data_evidence"):
        op.drop_column(table, "schema_version")
