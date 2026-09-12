"""Reusable authentication and ownership dependencies."""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AuthenticationError, NotFoundError
from app.db.models import Conversation
from app.db.session import get_session
from app.repositories.conversation import ConversationRepository
from app.schemas.auth import UserResponse
from app.services.auth import AuthService

bearer = HTTPBearer(auto_error=False)


def get_auth_service(request: Request) -> AuthService:
    """Resolve the app-owned authentication service."""
    service: AuthService = request.app.state.auth
    return service


async def get_current_user(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> UserResponse:
    """Require an access token and a currently active database identity."""
    if credentials is None:
        raise AuthenticationError()
    claims = service.codec.decode(credentials.credentials, "access")
    user = await service.current_user(claims)
    request.state.user_id = user.id
    structlog.contextvars.bind_contextvars(user_id=str(user.id))
    return user


async def get_owned_conversation(
    conversation_id: UUID,
    user: Annotated[UserResponse, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Conversation:
    """Hide missing, archived and foreign conversations behind the same 404."""
    conversation = await ConversationRepository(session, user.id).get(conversation_id)
    if conversation is None:
        raise NotFoundError()
    return conversation
