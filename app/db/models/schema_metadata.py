"""Application-owned reference data; these rows are global, not user-owned."""

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SchemaTableRecord(Base):
    """One description and notes block per physical business table."""

    __tablename__ = "schema_tables"

    table_name: Mapped[str] = mapped_column(String(127), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[list[str]] = mapped_column(JSONB, nullable=False)


class SchemaMetadataRecord(Base):
    """Authored semantic meaning with explicit expected physical properties."""

    __tablename__ = "schema_metadata"
    __table_args__ = (
        CheckConstraint(
            "semantic_type IN ('identifier','money','timestamp','enum','quantity','text','foreign_key')",
            name="semantic_type",
        ),
        CheckConstraint("(fk_table IS NULL) = (fk_column IS NULL)", name="fk_pair"),
        CheckConstraint("ordinal_position > 0", name="ordinal_position"),
    )

    table_name: Mapped[str] = mapped_column(
        String(127), ForeignKey("schema_tables.table_name", ondelete="RESTRICT"), primary_key=True
    )
    column_name: Mapped[str] = mapped_column(String(63), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    semantic_type: Mapped[str] = mapped_column(String(16), nullable=False)
    sql_type: Mapped[str] = mapped_column(String(200), nullable=False)
    ordinal_position: Mapped[int] = mapped_column(Integer, nullable=False)
    nullable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_primary_key: Mapped[bool] = mapped_column(Boolean, nullable=False)
    allowed_values: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    is_pii: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sample_values: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    fk_table: Mapped[str | None] = mapped_column(String(127))
    fk_column: Mapped[str | None] = mapped_column(String(63))
    notes: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    constraints: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
