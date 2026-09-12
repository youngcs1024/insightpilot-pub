"""Thin owned conversation and historical evidence adapters."""

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from app.agents.contracts import TurnIdentity
from app.api.dependencies import get_current_user
from app.api.v1.auth import Quota
from app.schemas.auth import UserResponse
from app.schemas.chat import (
    ConversationCreate,
    ConversationPage,
    ConversationResponse,
    EvidenceResponse,
    TurnPage,
)
from app.services.conversations import ConversationService

if TYPE_CHECKING:
    from app.services.evidence import EvidenceService

router = APIRouter(prefix="/conversations", tags=["conversations"])
User = Annotated[UserResponse, Depends(get_current_user)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


def get_conversations(request: Request) -> ConversationService:
    """Resolve the app-owned service."""
    service: ConversationService = request.app.state.conversations
    return service


Service = Annotated[ConversationService, Depends(get_conversations)]


async def conversation_user(request: Request, user: User, quota: Quota) -> UserResponse:
    """Apply every configured user quota to every conversation route."""
    quota.check("conversations", request, user.id)
    return user


OwnedUser = Annotated[UserResponse, Depends(conversation_user)]


@router.post("", status_code=201)
async def create(
    request: Request,
    user: OwnedUser,
    service: Service,
    body: ConversationCreate | None = None,
) -> ConversationResponse:
    """Create an explicitly titled or untitled conversation."""
    return await service.create(user.id, body.title if body is not None else "")


@router.get("")
async def page(
    request: Request, user: OwnedUser, service: Service, limit: Limit = 20, offset: Offset = 0
) -> ConversationPage:
    """List active conversations belonging to this user."""
    return await service.page(user.id, limit, offset)


@router.get("/{conversation_id}")
async def get(
    request: Request, conversation_id: UUID, user: OwnedUser, service: Service
) -> ConversationResponse:
    """Read an active owned conversation."""
    return await service.get(user.id, conversation_id)


@router.delete("/{conversation_id}")
async def archive(
    request: Request, conversation_id: UUID, user: OwnedUser, service: Service
) -> ConversationResponse:
    """Archive an idle conversation without deleting history."""
    return await service.archive(user.id, conversation_id)


@router.get("/{conversation_id}/turns")
async def turns(  # noqa: PLR0913, PLR0917 -- explicit FastAPI inputs.
    request: Request,
    conversation_id: UUID,
    user: OwnedUser,
    service: Service,
    limit: Limit = 20,
    offset: Offset = 0,
) -> TurnPage:
    """Read durable ordered messages."""
    return await service.turns(user.id, conversation_id, limit, offset)


@router.get("/{conversation_id}/turns/{turn_id}/evidence")
async def evidence(
    request: Request, conversation_id: UUID, turn_id: UUID, user: OwnedUser
) -> EvidenceResponse:
    """Read immutable snapshots with no external calls."""
    service: EvidenceService = request.app.state.evidence
    snapshot = await service.find(
        TurnIdentity(user_id=user.id, conversation_id=conversation_id, turn_id=turn_id)
    )
    return EvidenceResponse(turn_id=turn_id, data=snapshot)
