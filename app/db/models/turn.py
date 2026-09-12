"""Ordered application messages, independent of LangGraph checkpoint storage."""

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class TurnRole(StrEnum):
    """Only user inputs and assistant results belong in application history."""

    USER = "user"
    ASSISTANT = "assistant"


class TurnStatus(StrEnum):
    """Persisted lifecycle states; failures are never inferred from prose."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"
    ABSTAINED = "abstained"


class Turn(TimestampMixin, Base):
    """A message with database-enforced ordering, replay keys and execution exclusivity."""

    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uq_turns_conversation_id_seq"),
        Index(
            "uq_turns_conversation_id_idempotency_key",
            "conversation_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        Index(
            "uq_turns_conversation_id_running",
            "conversation_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    reply_to_turn_id: Mapped[UUID | None] = mapped_column(ForeignKey("turns.id"))
    seq: Mapped[int] = mapped_column(Integer())
    role: Mapped[TurnRole] = mapped_column(
        Enum(
            TurnRole, name="turn_role", values_callable=lambda roles: [role.value for role in roles]
        )
    )
    content: Mapped[str] = mapped_column(Text())
    status: Mapped[TurnStatus] = mapped_column(
        Enum(
            TurnStatus, name="turn_status", values_callable=lambda states: [s.value for s in states]
        )
    )
    route: Mapped[str | None] = mapped_column(String(32))
    failure_reason: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[int | None] = mapped_column(Integer())
    token_usage: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB(none_as_null=True))
    trace_id: Mapped[str | None] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))

    clarification: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB(none_as_null=True))
    answer: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB(none_as_null=True))
