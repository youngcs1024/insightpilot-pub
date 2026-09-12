"""Persist single-use refresh identifiers independently of API process lifetime."""

import sqlalchemy as sa

from alembic import op

revision: str = "0003_refresh_tokens"
down_revision: str | None = "0002_turn_reply"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create user-scoped refresh storage without changing existing users."""
    op.create_table(
        "refresh_tokens",
        sa.Column("jti_digest", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("jti_digest"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])


def downgrade() -> None:
    """Remove refresh storage when explicitly rolling back this migration."""
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
