"""Identity lookup is the only pre-authentication repository entrypoint."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.schemas.auth import RegisterRequest


class UserRepository:
    """Look up credentials by email; authenticated profile reads require user ID."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def by_email(self, email: str) -> User | None:
        """Resolve the case-insensitive login identity."""
        result: User | None = await self.session.scalar(select(User).where(User.email == email))
        return result

    async def get(self, user_id: UUID) -> User | None:
        """Read only the authenticated identity."""
        result: User | None = await self.session.scalar(select(User).where(User.id == user_id))
        return result

    async def create(self, request: RegisterRequest, hashed_password: str) -> User:
        """Insert without committing; database uniqueness arbitrates races."""
        user = User(
            email=str(request.email),
            display_name=request.display_name,
            hashed_password=hashed_password,
        )
        self.session.add(user)
        await self.session.flush()
        return user
