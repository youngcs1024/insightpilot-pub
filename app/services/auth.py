"""Authentication transactions and durable refresh rotation."""

from datetime import UTC, datetime
from uuid import UUID

import structlog

from app.core.config_models import SecuritySettings
from app.core.errors import AuthenticationError
from app.core.security import TokenCodec, hash_password, verify_password
from app.db.session import Database
from app.repositories.refresh_token import RefreshTokenRepository
from app.repositories.user import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenClaims, TokenResponse, UserResponse

logger = structlog.get_logger(__name__)
# A valid hash for a dummy comparison when an email does not exist.
DUMMY_HASH = "$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUW"


class AuthService:
    """Own transactions and expose typed, credential-free identity results."""

    def __init__(self, database: Database, settings: SecuritySettings) -> None:
        self.database = database
        self.settings = settings
        self.codec = TokenCodec(settings)

    async def register(self, body: RegisterRequest) -> UserResponse:
        """Hash before opening a transaction; uniqueness remains database enforced."""
        hashed = await hash_password(body.password.get_secret_value(), self.settings.bcrypt_rounds)
        async with self.database.session() as session, session.begin():
            user = await UserRepository(session).create(body, hashed)
            result = UserResponse.model_validate(user)
        logger.info("user_registered", user_id=str(result.id))
        return result

    async def authenticate(self, body: LoginRequest) -> UserResponse:
        """Use the same public failure for unknown, disabled or incorrect credentials."""
        async with self.database.session() as session:
            user = await UserRepository(session).by_email(str(body.email))
        valid = await verify_password(
            body.password.get_secret_value(), user.hashed_password if user else DUMMY_HASH
        )
        if not valid or user is None or not user.is_active:
            raise AuthenticationError()
        return UserResponse.model_validate(user)

    async def current_user(self, claims: TokenClaims) -> UserResponse:
        """Reload identity so disabling an account invalidates existing JWT access."""
        async with self.database.session() as session:
            user = await UserRepository(session).get(claims.sub)
            if user is None or not user.is_active:
                raise AuthenticationError()
            return UserResponse.model_validate(user)

    async def issue(self, user_id: UUID, prior: TokenClaims | None = None) -> TokenResponse:
        """Commit one atomic consume/issue transaction before returning bearer tokens."""
        access, access_claims = self.codec.create(user_id, "access")
        refresh, refresh_claims = self.codec.create(user_id, "refresh")
        async with self.database.session() as session, session.begin():
            user = await UserRepository(session).get(user_id)
            if user is None or not user.is_active:
                raise AuthenticationError()
            repository = RefreshTokenRepository(session, user_id)
            if prior is not None and (
                prior.sub != user_id
                or prior.typ != "refresh"
                or not await repository.consume(prior)
            ):
                raise AuthenticationError()
            await repository.add(refresh_claims)
        logger.info("refresh_rotated" if prior else "user_logged_in", user_id=str(user_id))
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            access_expires_at=datetime.fromtimestamp(access_claims.exp, UTC),
            refresh_expires_at=datetime.fromtimestamp(refresh_claims.exp, UTC),
        )
