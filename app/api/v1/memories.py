"""Authenticated, rate-limited memory history queries."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies import get_current_user
from app.api.v1.auth import Quota
from app.schemas.auth import UserResponse
from app.schemas.memory import MemoryType
from app.schemas.memory_api import MemoryPage
from app.services.memory.service import MemoryService

router = APIRouter(prefix="/memories", tags=["memories"])
User = Annotated[UserResponse, Depends(get_current_user)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


def get_memories(request: Request) -> MemoryService:
    """Resolve the application-owned history service."""
    service: MemoryService = request.app.state.memories
    return service


async def memory_user(request: Request, user: User, quota: Quota) -> UserResponse:
    """Apply every configured memory quota after authentication."""
    quota.check("memories", request, user.id)
    return user


OwnedUser = Annotated[UserResponse, Depends(memory_user)]
Service = Annotated[MemoryService, Depends(get_memories)]


@router.get("")
async def page(  # noqa: PLR0913, PLR0917 -- explicit FastAPI query inputs.
    request: Request,
    user: OwnedUser,
    service: Service,
    include_superseded: bool = False,
    memory_type: MemoryType | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> MemoryPage:
    """List active memories, or traverse preserved versions through their pointers."""
    return await service.page(
        user.id,
        include_superseded=include_superseded,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
