"""User-owned memories with immutable content and explicit supersession history."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint, Computed, DateTime, Enum, ForeignKey, Index, String, Text,
    UniqueConstraint, true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UserOwnedMixin
from app.schemas.memory import MemoryType


class MemoryRecord(UserOwnedMixin, TimestampMixin, Base):
    """Only lifecycle columns may be updated by the runtime database role."""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_user_type_active", "user_id", "memory_type", "is_active"),
        UniqueConstraint(
            "user_id", "memory_type", "active_metric_key",
            name="uq_memories_active_metric", deferrable=True, initially="DEFERRED",
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint(
            "(is_active AND superseded_by IS NULL AND superseded_at IS NULL) OR "
            "(NOT is_active AND superseded_by IS NOT NULL AND superseded_at IS NOT NULL)",
            name="supersession_state",
        ),
        CheckConstraint("superseded_by IS NULL OR superseded_by <> id", name="no_self_supersede"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(
            MemoryType,
            name="memory_type",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda kinds: [kind.value for kind in kinds],
        )
    )
    content: Mapped[dict[str, JsonValue]] = mapped_column(JSONB())
    summary: Mapped[str] = mapped_column(String(200))
    source_turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"))
    confidence: Mapped[float]
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    superseded_by: Mapped[UUID | None] = mapped_column(ForeignKey("memories.id"))
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active_metric_key: Mapped[str | None] = mapped_column(
        Text(),
        Computed(
            "CASE WHEN is_active AND memory_type = 'metric_override' "
            "THEN content ->> 'metric_key' ELSE NULL END",
            persisted=True,
        ),
    )
