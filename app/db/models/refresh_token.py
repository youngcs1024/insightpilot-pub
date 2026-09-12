"""Durable single-use refresh credentials; raw tokens are never stored."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UserOwnedMixin


class RefreshToken(UserOwnedMixin, Base):
    """One issued refresh identifier, bound to its authenticated user."""

    __tablename__ = "refresh_tokens"
    jti_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
