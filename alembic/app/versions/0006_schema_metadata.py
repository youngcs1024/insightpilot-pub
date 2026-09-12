"""Persist frozen Step 2.2 table and column semantics in the application database."""

import json
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import context, op

revision = "0006_schema_metadata"
down_revision = "0005_turn_answer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create normalized catalog tables and atomically load an immutable snapshot."""
    tables = op.create_table(
        "schema_tables",
        sa.Column("table_name", sa.String(127), primary_key=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("notes", JSONB, nullable=False),
    )
    columns = op.create_table(
        "schema_metadata",
        sa.Column(
            "table_name",
            sa.String(127),
            sa.ForeignKey("schema_tables.table_name", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("column_name", sa.String(63), primary_key=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("semantic_type", sa.String(16), nullable=False),
        sa.Column("sql_type", sa.String(200), nullable=False),
        sa.Column("ordinal_position", sa.Integer, nullable=False),
        sa.Column("nullable", sa.Boolean, nullable=False),
        sa.Column("is_primary_key", sa.Boolean, nullable=False),
        sa.Column("allowed_values", JSONB, nullable=False),
        sa.Column("is_pii", sa.Boolean, nullable=False),
        sa.Column("sample_values", JSONB, nullable=False),
        sa.Column("fk_table", sa.String(127)),
        sa.Column("fk_column", sa.String(63)),
        sa.Column("notes", JSONB, nullable=False),
        sa.Column("constraints", JSONB, nullable=False),
        sa.CheckConstraint(
            "semantic_type IN ('identifier','money','timestamp','enum','quantity','text','foreign_key')",
            name="semantic_type",
        ),
        sa.CheckConstraint("(fk_table IS NULL) = (fk_column IS NULL)", name="fk_pair"),
        sa.CheckConstraint("ordinal_position > 0", name="ordinal_position"),
    )
    snapshot = json.loads(
        (Path(__file__).parent.parent / "data/0006_schema_metadata.json").read_text()
    )
    table_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for table in snapshot["tables"]:
        table_rows.append({key: table[key] for key in ("table_name", "description", "notes")})
        column_rows.extend(
            {"table_name": table["table_name"], **column} for column in table["columns"]
        )
    insert_rows(tables, table_rows)
    insert_rows(columns, column_rows)


def insert_rows(table: sa.Table, rows: list[dict[str, object]]) -> None:
    """Support offline SQL output without a JSONB literal-rendering dependency."""
    if not context.is_offline_mode():
        op.bulk_insert(table, rows)
        return
    for row in rows:
        values = {
            key: sa.cast(sa.literal(json.dumps(value), sa.Text()), JSONB)
            if isinstance(table.c[key].type, JSONB)
            else value
            for key, value in row.items()
        }
        op.execute(table.insert().values(**values))


def downgrade() -> None:
    """Reverse only the two catalog tables introduced by this revision."""
    op.drop_table("schema_metadata")
    op.drop_table("schema_tables")
