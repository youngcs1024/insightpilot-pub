"""Synchronous and SSE adapters sharing one request-scoped lifecycle owner."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import structlog
from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse

from app.agents.runtime import RuntimeContext
from app.api.streaming import ChatStreamingResponse
from app.api.v1.auth import Quota
from app.api.v1.conversations import User
from app.core.deadline import Deadline, get_deadline
from app.db.models import TurnStatus
from app.schemas.chat import MessageRequest, TurnResponse
from app.services.chat import AdmittedTurn, ChatService
from app.services.chat_stream import stream
from app.services.idempotency import MessageAdmission
from app.services.regions import RegionService

router = APIRouter(prefix="/conversations", tags=["chat"])
Budget = Annotated[Deadline, Depends(get_deadline)]
Key = Annotated[str | None, Header(min_length=1, max_length=128)]


def get_chat(request: Request) -> ChatService:
    """Resolve the application lifecycle service."""
    service: ChatService = request.app.state.chat
    return service


Service = Annotated[ChatService, Depends(get_chat)]


async def admitted(  # noqa: PLR0913, PLR0917 -- typed dependency inputs.
    request: Request,
    conversation_id: UUID,
    body: MessageRequest,
    user: User,
    service: Service,
    quota: Quota,
    deadline: Budget,
    idempotency_key: Key = None,
) -> AsyncIterator[AdmittedTurn]:
    """Yield cleanup surrounds the entire response, including stream disconnects."""
    quota.check("messages", request, user.id)
    request.state.conversation_id = conversation_id
    structlog.contextvars.bind_contextvars(conversation_id=str(conversation_id))
    admission = MessageAdmission(
        user_id=user.id,
        conversation_id=conversation_id,
        content=body.content,
        idempotency_key=idempotency_key,
        trace_id=correlation_id.get(),
    )
    async with service.open(admission, deadline) as claim:
        structlog.contextvars.bind_contextvars(turn_id=str(claim.identity.turn_id))
        yield claim


Claim = Annotated[AdmittedTurn, Depends(admitted, scope="request")]


def runtime(request: Request, claim: AdmittedTurn, deadline: Deadline) -> RuntimeContext:
    """Build ephemeral runtime dependencies without passing request/session objects."""
    return RuntimeContext(
        regions=RegionService(request.app.state.mcp),
        metrics=request.app.state.metrics,
        clarification_capabilities=request.app.state.clarification_capabilities,
        now=datetime.now(UTC),
        schema_catalog=request.app.state.schema_catalog,
        schema_token_counter=request.app.state.schema_token_counter,
        llm=request.app.state.llm,
        retrieval=request.app.state.retrieval,
        knowledge_generation=request.app.state.knowledge_generation,
        mcp=request.app.state.mcp,
        evidence=request.app.state.evidence,
        conversations=request.app.state.conversations,
        settings=request.app.state.settings,
        deadline=deadline,
        identity=claim.identity,
        trace_id=claim.result.trace_id or correlation_id.get() or "",
    )


@router.post("/{conversation_id}/messages")
async def message(
    request: Request, response: Response, claim: Claim, service: Service, deadline: Budget
) -> TurnResponse:
    """Return the persisted assistant result; a running replay uses HTTP 202."""
    result = await service.execute(claim, runtime(request, claim, deadline))
    if result.replayed:
        response.headers["Idempotency-Replayed"] = "true"
    if result.status == TurnStatus.RUNNING:
        response.status_code = 202
    return result


@router.post("/{conversation_id}/messages/stream", response_model=None)
async def message_stream(
    request: Request, claim: Claim, service: Service, deadline: Budget
) -> Response:
    """Emit committed answer deltas or report an already-running original as JSON."""
    headers = {"Idempotency-Replayed": "true"} if claim.result.replayed else {}
    if claim.result.replayed and claim.result.status == TurnStatus.RUNNING:
        return JSONResponse(claim.result.model_dump(mode="json"), status_code=202, headers=headers)
    request.state.chat_stream = True
    headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return ChatStreamingResponse(
        stream(service, claim, runtime(request, claim, deadline)),
        headers=headers,
    )
