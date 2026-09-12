"""Append-only global metric versions, published exclusively by migrations."""

from sqlalchemy import Boolean, CheckConstraint, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class MetricDefinitionRecord(Base, TimestampMixin):
    """The active pointer can change while historical content remains available."""

    __tablename__ = "metric_definitions"
    __table_args__ = (
        CheckConstraint("version > 0", name="positive_version"),
        Index(
            "uq_metric_definitions_active_key",
            "key",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    expression_template: Mapped[str] = mapped_column(Text, nullable=False)
    base_tables: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    default_date_field: Mapped[str] = mapped_column(String(64), nullable=False)
    required_filters: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    supported_grains: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    examples: Mapped[list[dict[str, str]]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
