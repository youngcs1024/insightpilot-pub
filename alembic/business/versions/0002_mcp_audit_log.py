"""Add the MCP-owned audit table without granting read access to its writer."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002_mcp_audit_log"
down_revision = "0001_business_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the append-only audit storage in the bootstrap-owned mcp schema."""
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("caller", sa.Text(), nullable=False),
        sa.Column("correlation_id", sa.Text()),
        sa.Column("tool", sa.Text(), nullable=False),
        sa.Column("arguments_sha256", sa.Text(), nullable=False),
        sa.Column("sql_text", sa.Text()),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("reject_reasons", JSONB()),
        sa.Column("rows_returned", sa.Integer()),
        sa.Column("duration_ms", sa.Integer()),
        sa.CheckConstraint(
            "outcome IN ('ok','policy_rejected','execution_error','timeout')",
            name="ck_audit_log_outcome",
        ),
        schema="mcp",
    )
    op.create_index("ix_audit_occurred", "audit_log", [sa.text("occurred_at DESC")], schema="mcp")
    op.create_index(
        "ix_audit_outcome",
        "audit_log",
        ["outcome"],
        schema="mcp",
        postgresql_where=sa.text("outcome <> 'ok'"),
    )
    op.create_index("ix_audit_corr", "audit_log", ["correlation_id"], schema="mcp")
    op.execute("REVOKE ALL ON mcp.audit_log FROM PUBLIC, app_rw, etl_rw, mcp_ro, mcp_audit")
    op.execute(
        "REVOKE ALL ON SEQUENCE mcp.audit_log_id_seq FROM PUBLIC, app_rw, etl_rw, mcp_ro, mcp_audit"
    )
    op.execute("GRANT INSERT ON mcp.audit_log TO mcp_audit")
    op.execute("GRANT USAGE ON SEQUENCE mcp.audit_log_id_seq TO mcp_audit")


def downgrade() -> None:
    """Remove this revision's audit table; bootstrap continues to own its schema."""
    op.drop_index("ix_audit_corr", table_name="audit_log", schema="mcp")
    op.drop_index("ix_audit_outcome", table_name="audit_log", schema="mcp")
    op.drop_index("ix_audit_occurred", table_name="audit_log", schema="mcp")
    op.drop_table("audit_log", schema="mcp")
