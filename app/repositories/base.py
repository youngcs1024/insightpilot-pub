"""A bound user identity is required before constructing an owned-data query."""

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import UserOwnedMixin


class UserScopedRepository[ModelT: UserOwnedMixin]:
    """Start every owned-data query from _scoped(); never commit in a repository."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession, user_id: UUID) -> None:
        self._session = session
        self._user_id = user_id

    def _scoped(self) -> Select[tuple[ModelT]]:
        return select(self.model).where(self.model.user_id == self._user_id)
