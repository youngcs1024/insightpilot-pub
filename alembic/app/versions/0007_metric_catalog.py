"""Publish six immutable metric definitions using a frozen catalog snapshot."""

import json
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import context, op

revision = "0007_metric_catalog"
down_revision = "0006_schema_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Atomically create, seed and grant read-only runtime access to the catalog."""
    table = op.create_table(
        "metric_definitions",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("version", sa.Integer, primary_key=True),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("expression_template", sa.Text, nullable=False),
        sa.Column("base_tables", JSONB, nullable=False),
        sa.Column("default_date_field", sa.String(64), nullable=False),
        sa.Column("required_filters", JSONB, nullable=False),
        sa.Column("supported_grains", JSONB, nullable=False),
        sa.Column("examples", JSONB, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("version > 0", name="positive_version"),
    )
    op.create_index(
        "uq_metric_definitions_active_key",
        "metric_definitions",
        ["key"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )
    snapshot = json.loads(
        (Path(__file__).parent.parent / "data/0007_metric_catalog.json").read_text(encoding="utf-8")
    )
    if context.is_offline_mode():
        for row in snapshot["definitions"]:
            values = {
                key: sa.cast(sa.literal(json.dumps(value), sa.Text()), JSONB)
                if isinstance(table.c[key].type, JSONB)
                else value
                for key, value in row.items()
            }
            op.execute(table.insert().values(**values))
    else:
        op.bulk_insert(table, snapshot["definitions"])
    op.execute("REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON metric_definitions FROM app_rw")
    op.execute("GRANT SELECT ON metric_definitions TO app_rw")


def downgrade() -> None:
    """Remove only the table owned by this revision."""
    op.drop_table("metric_definitions")
