"""Bounded authentication contracts; secrets never appear in model reprs."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from asgi_correlation_id import correlation_id
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


def current_request_id() -> str | None:
    """Read the edge correlation context for each response."""
    return correlation_id.get()


class AuthResponse(BaseModel):
    """Request-correlated response base."""

    request_id: str | None = Field(default_factory=current_request_id)


class RegisterRequest(BaseModel):
    """New identity; password policy is business validation."""

    email: EmailStr = Field(max_length=254)
    password: SecretStr = Field(max_length=1024)
    display_name: str = Field(min_length=1, max_length=100)


class LoginRequest(BaseModel):
    """JSON credentials."""

    email: EmailStr = Field(max_length=254)
    password: SecretStr = Field(max_length=1024)


class RefreshRequest(BaseModel):
    """A single refresh credential."""

    refresh_token: SecretStr = Field(min_length=1, max_length=4096)


class UserResponse(AuthResponse):
    """Public profile without credentials."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    display_name: str
    is_active: bool


class TokenResponse(AuthResponse):
    """An access/refresh pair returned only after database commit."""

    access_token: str = Field(repr=False)
    refresh_token: str = Field(repr=False)
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 -- OAuth scheme, not a secret.
    access_expires_at: datetime
    refresh_expires_at: datetime


class TokenClaims(BaseModel):
    """Validated JWT payload."""

    sub: UUID
    typ: Literal["access", "refresh"]
    exp: int = Field(strict=True)
    iat: int = Field(strict=True)
    jti: str = Field(min_length=16, max_length=128)
