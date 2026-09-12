"""Fixed-algorithm JWTs and bounded, off-thread password hashing."""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

import bcrypt
import jwt
from pydantic import ValidationError as ModelValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config_models import SecuritySettings
from app.core.errors import AuthenticationError, PasswordPolicyError, TokenExpiredError
from app.schemas.auth import TokenClaims

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72


def password_failures(password: str) -> list[str]:
    """List every failed rule without including any supplied input."""
    checks = (
        (len(password) >= MIN_PASSWORD_LENGTH, "at least 8 characters"),
        (len(password.encode("utf-8")) <= MAX_PASSWORD_BYTES, "at most 72 UTF-8 bytes"),
        (any(c.isupper() for c in password), "an uppercase letter"),
        (any(c.islower() for c in password), "a lowercase letter"),
        (any(c.isdecimal() for c in password), "a digit"),
        (any(not c.isalnum() and not c.isspace() for c in password), "a symbol"),
    )
    return [rule for valid, rule in checks if not valid]


async def hash_password(password: str, rounds: int) -> str:
    """Validate policy and move CPU work off the event loop."""
    failures = password_failures(password)
    if failures:
        raise PasswordPolicyError(failures)
    hashed = await run_in_threadpool(bcrypt.hashpw, password.encode(), bcrypt.gensalt(rounds))
    return hashed.decode()


async def verify_password(password: str, hashed: str) -> bool:
    """Reject excessive inputs rather than letting bcrypt truncate them."""
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        return False
    try:
        return await run_in_threadpool(bcrypt.checkpw, password.encode(), hashed.encode())
    except ValueError:
        return False


class TokenCodec:
    """Use one injected signing configuration per application."""

    def __init__(self, settings: SecuritySettings) -> None:
        self.settings = settings

    def create(self, subject: UUID, typ: Literal["access", "refresh"]) -> tuple[str, TokenClaims]:
        """Mint random token identifiers with explicit purpose and expiry."""
        now = datetime.now(UTC)
        ttl = (
            timedelta(minutes=self.settings.access_ttl_minutes)
            if typ == "access"
            else timedelta(days=self.settings.refresh_ttl_days)
        )
        claims = TokenClaims(
            sub=subject,
            typ=typ,
            exp=int((now + ttl).timestamp()),
            iat=int(now.timestamp()),
            jti=secrets.token_urlsafe(24),
        )
        token = jwt.encode(
            claims.model_dump(mode="json"),
            self.settings.jwt_secret.get_secret_value(),
            algorithm="HS256",
        )
        return token, claims

    def decode(self, token: str, expect_typ: Literal["access", "refresh"]) -> TokenClaims:
        """Verify cryptography and typed claims before trusting identity."""
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                options={"require": ["sub", "typ", "exp", "iat", "jti"]},
            )
            claims = TokenClaims.model_validate(payload)
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError() from exc
        except (jwt.InvalidTokenError, ModelValidationError, TypeError, OverflowError) as exc:
            raise AuthenticationError() from exc
        if claims.typ != expect_typ:
            raise AuthenticationError()
        return claims
