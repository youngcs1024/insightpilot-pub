"""Associate an assistant outcome with the exact original user message."""

import sqlalchemy as sa

from alembic import op

revision: str = "0002_turn_reply"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Preserve existing turns; new admissions populate the explicit association."""
    op.add_column("turns", sa.Column("reply_to_turn_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_turns_reply_to_turn_id_turns", "turns", "turns", ["reply_to_turn_id"], ["id"]
    )


def downgrade() -> None:
    """Restore the original turn schema without changing message rows."""
    op.drop_constraint("fk_turns_reply_to_turn_id_turns", "turns", type_="foreignkey")
    op.drop_column("turns", "reply_to_turn_id")
