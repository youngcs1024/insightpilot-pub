"""Heartbeat and cancellation orchestration over committed answer deltas."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Literal

import anyio
from pydantic import BaseModel

from app.agents.failures import FailureKind
from app.agents.runtime import RuntimeContext
from app.core.background import spawn
from app.core.errors import InsightPilotError
from app.db.models import TurnStatus
from app.schemas.chat import ErrorEvent, HeartbeatEvent, TokenEvent
from app.services.chat import AdmittedTurn, ChatService
from app.services.turn_results import TurnFailedError

HEARTBEAT_SECONDS = 15.0
CHUNK_CHARACTERS = 128


def frame(event: Literal["token", "heartbeat", "error", "done"], payload: BaseModel) -> str:
    """JSON escapes line breaks so user content cannot inject SSE frames."""
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"


def error_event(error: InsightPilotError) -> ErrorEvent:
    """Never serialize internal diagnostics."""
    code = error.reason.value if isinstance(error, TurnFailedError) else error.code
    return ErrorEvent(code=code, message=error.user_message)


async def stream(
    service: ChatService, claim: AdmittedTurn, ctx: RuntimeContext
) -> AsyncGenerator[str, None]:
    """Execute once, emit keepalives, then release the persisted answer."""
    task = spawn(service.execute(claim, ctx), name="chat-graph")
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
            if not done:
                yield frame("heartbeat", HeartbeatEvent())
        result = task.result()
        if result.status == TurnStatus.FAILED:
            raise TurnFailedError(result.failure_reason or FailureKind.NODE_OPERATION_FAILED)
        content = (
            result.content
            if result.clarification is not None
            else (result.answer.markdown if result.answer is not None else "")
        )
        for offset in range(0, len(content), CHUNK_CHARACTERS):
            yield frame("token", TokenEvent(delta=content[offset : offset + CHUNK_CHARACTERS]))
        yield frame("done", result)
    except InsightPilotError as exc:
        yield frame("error", error_event(exc))
    finally:
        task.cancel()
        with anyio.CancelScope(shield=True):
            with suppress(asyncio.CancelledError, InsightPilotError):
                await asyncio.shield(task)
