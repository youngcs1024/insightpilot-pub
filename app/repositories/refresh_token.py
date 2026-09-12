"""Atomic, user-scoped refresh consumption; the service owns commit."""

from datetime import UTC, datetime
from hashlib import sha256

from sqlalchemy import func, update

from app.core.errors import AuthenticationError
from app.db.models.refresh_token import RefreshToken
from app.repositories.base import UserScopedRepository
from app.schemas.auth import TokenClaims


class RefreshTokenRepository(UserScopedRepository[RefreshToken]):
    """Persist token identifiers without storing bearer credentials."""

    model = RefreshToken

    async def add(self, claims: TokenClaims) -> None:
        """Bind new credentials to the repository's user identity."""
        if claims.sub != self._user_id or claims.typ != "refresh":
            raise AuthenticationError()
        self._session.add(
            RefreshToken(
                jti_digest=sha256(claims.jti.encode()).hexdigest(),
                user_id=self._user_id,
                expires_at=datetime.fromtimestamp(claims.exp, UTC),
            )
        )
        await self._session.flush()

    async def consume(self, claims: TokenClaims) -> bool:
        """Only one concurrent transaction can consume an unexpired token."""
        consumed = await self._session.scalar(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == self._user_id,
                RefreshToken.jti_digest == sha256(claims.jti.encode()).hexdigest(),
                RefreshToken.used_at.is_(None),
                RefreshToken.expires_at > func.clock_timestamp(),
            )
            .values(used_at=func.clock_timestamp())
            .returning(RefreshToken.jti_digest)
        )
        return consumed is not None
