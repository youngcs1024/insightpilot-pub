"""Record a distinct MCP rate-limit outcome without rewriting existing audit rows."""

from alembic import op

revision = "0003_mcp_rate_limit_outcome"
down_revision = "0002_mcp_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Allow the write-only auditor to record quota denials distinctly."""
    op.drop_constraint("ck_audit_log_outcome", "audit_log", schema="mcp", type_="check")
    op.create_check_constraint(
        op.f("ck_audit_log_outcome"),
        "audit_log",
        "outcome IN ('ok','policy_rejected','execution_error','timeout','rate_limited')",
        schema="mcp",
    )


def downgrade() -> None:
    """Restore the former check only when no new outcome would be lost."""
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM mcp.audit_log WHERE outcome = 'rate_limited') THEN
                RAISE EXCEPTION 'Cannot downgrade while rate_limited audit rows exist';
            END IF;
        END $$
    """)
    op.drop_constraint("ck_audit_log_outcome", "audit_log", schema="mcp", type_="check")
    op.create_check_constraint(
        op.f("ck_audit_log_outcome"),
        "audit_log",
        "outcome IN ('ok','policy_rejected','execution_error','timeout')",
        schema="mcp",
    )
