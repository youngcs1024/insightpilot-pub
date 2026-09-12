"""Immutable application-owned data snapshots, independent of business source lifetime."""

from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class DataEvidenceRecord(TimestampMixin, Base):
    """One canonical snapshot per owned assistant turn."""

    __tablename__ = "data_evidence"
    __table_args__ = (UniqueConstraint("user_id", "assistant_turn_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    assistant_turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"))
    content_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, JsonValue]] = mapped_column(JSONB())
