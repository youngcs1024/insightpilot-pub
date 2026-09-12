"""JSON authentication routes; identity resolution precedes user quotas."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import bearer, get_auth_service, get_current_user
from app.core.errors import AuthenticationError
from app.core.limiter import AuthLimiter
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
Service = Annotated[AuthService, Depends(get_auth_service)]


def get_limiter(request: Request) -> AuthLimiter:
    """Resolve this application's isolated quota backend."""
    limiter: AuthLimiter = request.app.state.auth_limiter
    return limiter


Quota = Annotated[AuthLimiter, Depends(get_limiter)]


@router.post("/register", status_code=201)
async def register(
    request: Request, body: RegisterRequest, service: Service, quota: Quota
) -> UserResponse:
    """Register an identity without implicitly logging in."""
    quota.check("register", request)
    return await service.register(body)


@router.post("/login")
async def login(
    request: Request, body: LoginRequest, service: Service, quota: Quota
) -> TokenResponse:
    """Validate credentials and issue a durable refresh token."""
    try:
        user = await service.authenticate(body)
    except AuthenticationError:
        quota.check("login", request)
        raise
    quota.check("login", request, user.id)
    return await service.issue(user.id)


@router.post("/refresh")
async def refresh(
    request: Request, body: RefreshRequest, service: Service, quota: Quota
) -> TokenResponse:
    """Rotate a single-use refresh token."""
    try:
        claims = service.codec.decode(body.refresh_token.get_secret_value(), "refresh")
        user = await service.current_user(claims)
    except AuthenticationError:
        quota.check("refresh", request)
        raise
    quota.check("refresh", request, user.id)
    return await service.issue(user.id, claims)


async def get_me_user(
    request: Request,
    service: Service,
    quota: Quota,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> UserResponse:
    """Apply an IP quota to rejected credentials and a user quota to valid ones."""
    try:
        user = await get_current_user(request, service, credentials)
    except AuthenticationError:
        quota.check("me", request)
        raise
    quota.check("me", request, user.id)
    return user


@router.get("/me")
async def me(user: Annotated[UserResponse, Depends(get_me_user)]) -> UserResponse:
    """Read only the authenticated profile."""
    return user
