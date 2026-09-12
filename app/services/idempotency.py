"""Durable message admission and bounded failure finalization, without graph execution."""

import asyncio
from datetime import timedelta
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.agents.failures import FailureKind
from app.core.deadline import Deadline
from app.core.errors import ConflictError, DeadlineExceededError, NotFoundError
from app.db.models import Turn, TurnRole, TurnStatus
from app.db.session import Database
from app.repositories.turns import TurnRepository

logger = structlog.get_logger(__name__)
REPLAY_WINDOW = timedelta(hours=24)


class MessageAdmission(BaseModel):
    """Validated internal input; identity comes from the caller's trusted auth context."""

    user_id: UUID
    conversation_id: UUID
    content: str = Field(min_length=1, max_length=32_000)
    trace_id: str | None = Field(default=None, max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class AdmissionResult(BaseModel):
    """The original assistant outcome, not the associated user input."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    conversation_id: UUID
    reply_to_turn_id: UUID
    content: str
    status: TurnStatus
    failure_reason: str | None
    trace_id: str | None
    replayed: bool = False


class IdempotencyService:
    """Own short transactions; never execute or retry a graph from admission."""

    def __init__(self, database: Database, timeout_s: float) -> None:
        self.database = database
        self.timeout_s = timeout_s

    async def admit(self, request: MessageAdmission, *, deadline: Deadline) -> AdmissionResult:
        """Atomically replay or claim a new assistant, bounded by the request deadline."""
        deadline.check("message_admission")
        timer = asyncio.timeout(deadline.remaining())
        try:
            async with timer:
                return await self._admit(request, deadline=deadline)
        except TimeoutError as exc:
            if timer.expired():
                raise DeadlineExceededError(operation="message_admission") from exc
            raise

    async def _admit(self, request: MessageAdmission, *, deadline: Deadline) -> AdmissionResult:
        async with self.database.session() as session:
            async with asyncio.timeout(deadline.budget(self.timeout_s)), session.begin():
                repo = TurnRepository(session, request.user_id)
                await repo.lock_conversation(request.conversation_id)
                result = await self.admit_locked(repo, request)
            logger.info("turn_admitted", assistant_turn_id=str(result.id))
            return result

    async def admit_locked(
        self, repo: TurnRepository, request: MessageAdmission, *, allow_new: bool = True
    ) -> AdmissionResult:
        """Caller owns the conversation lock and transaction; replay precedes exclusivity."""
        replay = await self._find_replay(repo, request)
        if replay is not None:
            return replay
        if not allow_new:
            raise ConflictError()
        assistant = await repo.create_pair(
            request.conversation_id, request.content, request.idempotency_key
        )
        assistant.trace_id = request.trace_id
        return AdmissionResult.model_validate(assistant)

    async def _find_replay(
        self, repo: TurnRepository, request: MessageAdmission
    ) -> AdmissionResult | None:
        if request.idempotency_key is None:
            return None
        prior = await repo.find_key(request.conversation_id, request.idempotency_key)
        if prior is None:
            return None
        now = await repo.current_time()
        if prior.status == TurnStatus.RUNNING or now < prior.created_at + REPLAY_WINDOW:
            return await self._replay(repo, request, prior)
        await repo.release_key(request.conversation_id, prior.id)
        return None

    async def _replay(
        self, repo: TurnRepository, request: MessageAdmission, prior: Turn
    ) -> AdmissionResult:
        if prior.role != TurnRole.ASSISTANT or prior.reply_to_turn_id is None:
            raise ConflictError("legacy idempotency row lacks an assistant association")
        original = await repo.get(request.conversation_id, prior.reply_to_turn_id)
        if (
            original is None
            or original.role != TurnRole.USER
            or original.content != request.content
        ):
            raise ConflictError("idempotency key reused with different content")
        logger.info("turn_replayed", assistant_turn_id=str(prior.id))
        return AdmissionResult.model_validate(prior).model_copy(update={"replayed": True})

    async def fail_deadline(self, *, user_id: UUID, conversation_id: UUID, turn_id: UUID) -> None:
        """Finalize a running assistant with a fresh, bounded cleanup transaction.

        Call after unwinding the request timeout. This changes only lifecycle fields;
        committed evidence and already terminal outcomes are never overwritten.
        """
        async with self.database.session() as session:
            async with asyncio.timeout(self.timeout_s), session.begin():
                repo = TurnRepository(session, user_id)
                await repo.lock_conversation(conversation_id)
                turn = await repo.get(conversation_id, turn_id)
                if turn is None or turn.role != TurnRole.ASSISTANT:
                    raise NotFoundError()
                if turn.status != TurnStatus.RUNNING:
                    return
                turn.status = TurnStatus.FAILED
                turn.failure_reason = FailureKind.DEADLINE_EXCEEDED.value
            logger.info(
                "turn_failed",
                assistant_turn_id=str(turn_id),
                reason=FailureKind.DEADLINE_EXCEEDED.value,
            )
