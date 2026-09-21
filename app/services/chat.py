"""Own admission, execution, durable results and bounded cancellation cleanup."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from uuid import UUID

import anyio
import structlog
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.agents.answer_validation import validate_formatted_answer
from app.agents.contracts import Answer, EvidenceRefs, Route, TurnIdentity
from app.agents.failures import FailureKind
from app.agents.nodes.format_answer import attempted_sources
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphOutput
from app.core.background import spawn
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import ConflictError, DeadlineExceededError, InsightPilotError, NotFoundError
from app.core.observability import (
    GraphTraceCallback,
    Observability,
    Observation,
    TraceMetadata,
    record_turn_status,
)
from app.db.models import TurnStatus
from app.db.session import Database, translate_database_error
from app.repositories import lifecycle
from app.repositories.evidence import EvidenceRepository
from app.repositories.turns import TurnRepository
from app.schemas.chat import TurnResponse
from app.schemas.knowledge import KnowledgeDraft
from app.schemas.metric_resolution import MetricClarification
from app.services.conversations import ConversationService
from app.services.graph import GraphService
from app.services.idempotency import AdmissionResult, IdempotencyService, MessageAdmission
from app.services.knowledge_generation import validate_citations
from app.services.turn_results import (
    BothSourcesFailedError,
    TurnFailedError,
    failure_reason,
    turn_response,
)

logger = structlog.get_logger(__name__)


class AdmittedTurn(BaseModel):
    """An owned request claim; replay never owns the original execution."""

    identity: TurnIdentity
    result: TurnResponse


class ChatService:
    """Short application transactions surround asynchronous graph execution."""

    def __init__(self, database: Database, settings: Settings, graph: GraphService) -> None:
        self.database = database
        self.settings = settings
        self.graph = graph
        self.observability = Observability(settings)
        self.timeout_s = settings.database.command_timeout_s
        self._lease_slots = asyncio.Semaphore(
            max(1, settings.database.pool_size + settings.database.max_overflow - 1)
        )
        self.idempotency = IdempotencyService(database, self.timeout_s)
        self.conversations = ConversationService(database)

    @asynccontextmanager
    async def lease(self, conversation_id: UUID) -> AsyncIterator[bool]:
        """Keep a session lock without a long-running database transaction."""
        try:
            async with self._lease_slots, self.database.engine.connect() as connection:
                await connection.execution_options(isolation_level="AUTOCOMMIT")
                acquired: bool | None = None
                try:
                    acquired = await lifecycle.try_lease(connection, conversation_id)
                    yield acquired
                finally:
                    await self._finish_lease(connection, conversation_id, acquired)
        except (SQLAlchemyError, TimeoutError, OSError) as exc:
            raise translate_database_error(exc) from exc

    async def _finish_lease(
        self, connection: AsyncConnection, conversation_id: UUID, acquired: bool | None
    ) -> None:
        if acquired is None:
            await connection.invalidate()
        elif acquired:
            await self._release(connection, conversation_id)

    async def _release(self, connection: AsyncConnection, conversation_id: UUID) -> None:
        with anyio.CancelScope(shield=True):
            try:
                async with asyncio.timeout(self.timeout_s):
                    await lifecycle.release_lease(connection, conversation_id)
            except BaseException:
                await connection.invalidate()
                raise

    @asynccontextmanager
    async def open(
        self, request: MessageAdmission, deadline: Deadline
    ) -> AsyncIterator[AdmittedTurn]:
        """Claim before headers; retain the lease through response cleanup."""
        await self.conversations.get(request.user_id, request.conversation_id)
        async with self._claim(request, deadline) as admission:
            identity = TurnIdentity(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                turn_id=admission.id,
            )
            try:
                claim = AdmittedTurn(
                    identity=identity, result=await self.read(identity, replayed=admission.replayed)
                )
                yield claim
            finally:
                if not admission.replayed:
                    reason = (
                        FailureKind.DEADLINE_EXCEEDED
                        if deadline.remaining() <= 0
                        else FailureKind.CLIENT_DISCONNECTED
                    )
                    await self.cleanup(identity, reason)

    @asynccontextmanager
    async def _claim(
        self, request: MessageAdmission, deadline: Deadline
    ) -> AsyncIterator[AdmissionResult]:
        # Lock admission and execution ownership in one transaction: even simultaneous
        # same-key requests must observe the original committed pair, never a gap.
        try:
            async with self._lease_slots, self.database.engine.connect() as connection:
                acquired: bool | None = None
                try:
                    admission, acquired = await self._admit(connection, request, deadline)
                    yield admission
                finally:
                    await self._finish_lease(connection, request.conversation_id, acquired)
        except (SQLAlchemyError, TimeoutError, OSError) as exc:
            raise translate_database_error(exc) from exc

    async def _admit(
        self, connection: AsyncConnection, request: MessageAdmission, deadline: Deadline
    ) -> tuple[AdmissionResult, bool]:
        async with (
            AsyncSession(bind=connection, expire_on_commit=False) as session,
            asyncio.timeout(deadline.budget(self.timeout_s)),
            session.begin(),
        ):
            repo = TurnRepository(session, request.user_id)
            await repo.lock_conversation(request.conversation_id)
            acquired = await lifecycle.try_lease(connection, request.conversation_id)
            admission = await self.idempotency.admit_locked(repo, request, allow_new=acquired)
        return admission, acquired

    async def read(self, identity: TurnIdentity, *, replayed: bool = False) -> TurnResponse:
        """Read only persisted data for replay."""
        async with self.database.session() as session:
            row = await TurnRepository(session, identity.user_id).get(
                identity.conversation_id, identity.turn_id
            )
            if row is None:
                raise NotFoundError()
            return turn_response(row, replayed=replayed)

    async def execute(self, claim: AdmittedTurn, ctx: RuntimeContext) -> TurnResponse:
        """Commit evidence-validated answers before releasing any answer bytes."""
        if claim.result.replayed:
            return claim.result
        metadata = TraceMetadata(
            user_id=str(ctx.identity.user_id),
            conversation_id=str(ctx.identity.conversation_id),
            turn_id=str(ctx.identity.turn_id),
            request_id=ctx.trace_id,
            status="running",
        )
        with self.observability.turn(ctx.trace_id, metadata) as observation:
            try:
                result = await self._execute(ctx, observation)
            except BaseException:
                if observation.metadata.status not in {"succeeded", "degraded", "abstained"}:
                    observation.update(TraceMetadata(status="failed"))
                raise
            else:
                observation.update(TraceMetadata(status=result.status.value))
                return result

    async def _execute(self, ctx: RuntimeContext, observation: Observation) -> TurnResponse:
        started = monotonic()
        try:
            async with asyncio.timeout(ctx.deadline.remaining()):
                callback = GraphTraceCallback()
                try:
                    output = await self.graph.invoke(ctx, callbacks=[callback])
                finally:
                    callback.close()
                if output.status == "failed" or (
                    output.answer is None and output.clarification is None
                ):
                    reason = (
                        output.failures[-1].kind
                        if output.failures
                        else FailureKind.NODE_OPERATION_FAILED
                    )
                    both_sources = (
                        output.route is not None
                        and output.route.route is Route.BOTH
                        and (
                            output.evidence_refs is None
                            or (
                                output.evidence_refs.data_snapshot_id is None
                                and output.evidence_refs.knowledge_snapshot_id is None
                            )
                        )
                    )
                    error_type = BothSourcesFailedError if both_sources else TurnFailedError
                    raise error_type(reason)
                latency_ms = int((monotonic() - started) * 1000)
                if output.clarification is not None:
                    await self._clarify(ctx.identity, output, latency_ms)
                else:
                    await self._succeed(ctx.identity, output, latency_ms)
                observation.update(TraceMetadata(status=output.status))
                return await self.read(ctx.identity)
        except TimeoutError as exc:
            await self.cleanup(ctx.identity, FailureKind.DEADLINE_EXCEEDED)
            raise DeadlineExceededError() from exc
        except asyncio.CancelledError:
            reason = (
                FailureKind.DEADLINE_EXCEEDED
                if ctx.deadline.remaining() <= 0
                else FailureKind.CLIENT_DISCONNECTED
            )
            await self.cleanup(ctx.identity, reason)
            raise
        except InsightPilotError as exc:
            logger.exception(
                "turn_execution_failed", turn_id=str(ctx.identity.turn_id), code=exc.code
            )
            await self.cleanup(ctx.identity, failure_reason(exc))
            raise
        except Exception as exc:
            logger.exception("turn_execution_failed", turn_id=str(ctx.identity.turn_id))
            await self.cleanup(ctx.identity, FailureKind.NODE_OPERATION_FAILED)
            raise InsightPilotError() from exc

    async def _clarify(self, identity: TurnIdentity, output: GraphOutput, latency_ms: int) -> None:
        """Commit a typed request for information without inventing analysis evidence."""
        clarification = output.clarification
        if clarification is None or output.answer is None or output.data_evidence is not None:
            raise ConflictError("inconsistent clarification result")
        if (
            output.evidence_refs is not None
            and (
                output.evidence_refs.data_snapshot_id is not None
                or output.evidence_refs.knowledge_snapshot_id is not None
            )
        ) or output.failures:
            raise ConflictError("clarification contains analysis state")
        async with self.database.session() as session, session.begin():
            repo = TurnRepository(session, identity.user_id)
            await repo.lock_conversation(identity.conversation_id)
            row = await repo.get(identity.conversation_id, identity.turn_id)
            if row is None or row.status != TurnStatus.RUNNING:
                raise ConflictError()
            answer = output.answer
            if (
                answer.trace_id != row.trace_id
                or answer.evidence_refs != (output.evidence_refs or EvidenceRefs())
                or answer.attempted_sources
                or not answer.abstained
                or output.status != "abstained"
            ):
                raise ConflictError("clarification envelope differs from turn")
            await self._validate_answer(
                EvidenceRepository(session, identity), answer, clarification
            )
            row.clarification = clarification.model_dump(mode="json")
            row.content = answer.markdown
            row.answer = answer.model_dump(mode="json")
            row.status = TurnStatus(output.status)
            row.failure_reason = None
            row.latency_ms = latency_ms
        logger.info("turn_clarified", turn_id=str(identity.turn_id), kind=clarification.kind.value)

    async def _succeed(self, identity: TurnIdentity, output: GraphOutput, latency_ms: int) -> None:
        async with self.database.session() as session, session.begin():
            repo = TurnRepository(session, identity.user_id)
            await repo.lock_conversation(identity.conversation_id)
            row = await repo.get(identity.conversation_id, identity.turn_id)
            if row is None or row.status != TurnStatus.RUNNING:
                raise ConflictError()
            answer = output.answer
            if answer is None or output.evidence_refs != answer.evidence_refs:
                raise ConflictError()
            if (
                output.route is not None
                and output.route.route is Route.BOTH
                and answer.synthesis is None
                and (
                    answer.evidence_refs.data_snapshot_id
                    or answer.evidence_refs.knowledge_snapshot_id
                )
            ):
                raise ConflictError("BOTH answer requires synthesis")
            if (
                answer.trace_id != row.trace_id
                or answer.attempted_sources
                != attempted_sources(output.route.route if output.route else None)
                or (output.status == "abstained") != answer.abstained
                or (
                    not answer.abstained
                    and (output.status == "degraded") != bool(answer.degraded_components)
                )
            ):
                raise ConflictError("answer envelope differs from turn")
            await self._validate_answer(EvidenceRepository(session, identity), answer)
            row.answer = answer.model_dump(mode="json")
            row.content = answer.markdown
            row.status = TurnStatus(output.status)
            row.failure_reason = None
            row.latency_ms = latency_ms
        logger.info(
            "turn_completed",
            turn_id=str(identity.turn_id),
            status=output.status,
            latency_ms=latency_ms,
        )

    async def _validate_answer(
        self,
        repo: EvidenceRepository,
        answer: Answer,
        clarification: MetricClarification | None = None,
    ) -> None:
        bundle = await repo.read_bundle()
        if bundle.refs != answer.evidence_refs:
            raise ConflictError("answer references differ from snapshots")
        expected_sql = bundle.data.data.sql if bundle.data else ""
        expected_assumptions = list(
            dict.fromkeys(
                [
                    *(bundle.data.data.assumptions if bundle.data else []),
                    *(bundle.knowledge.knowledge.assumptions if bundle.knowledge else []),
                ]
            )
        )
        if answer.sql != expected_sql or answer.assumptions != expected_assumptions:
            raise ConflictError("answer fields differ from snapshots")
        validate_formatted_answer(answer, bundle, clarification)
        if answer.knowledge_passages:
            if bundle.knowledge is None:
                raise ConflictError("knowledge answer without snapshot")
            citations = validate_citations(
                KnowledgeDraft(passages=tuple(answer.knowledge_passages)),
                bundle.knowledge.knowledge,
            )
            if list(citations) != answer.citations:
                raise ConflictError("citation metadata differs from snapshot")
        elif answer.citations:
            raise ConflictError("citations without passages")
        if bundle.data is None and bundle.knowledge is None and not answer.abstained:
            raise ConflictError("analytical answer without evidence")

    async def cleanup(self, identity: TurnIdentity, reason: FailureKind) -> None:
        """Shield bounded cleanup from disconnect cancellation; preserve terminal outcomes."""
        with anyio.CancelScope(shield=True):
            task = spawn(self._fail(identity, reason), name="finalize-chat-turn")
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("turn_cleanup_failed", turn_id=str(identity.turn_id))

    async def _fail(self, identity: TurnIdentity, reason: FailureKind) -> None:
        async with (
            asyncio.timeout(self.timeout_s),
            self.database.session() as session,
            session.begin(),
        ):
            repo = TurnRepository(session, identity.user_id)
            await repo.lock_conversation(identity.conversation_id)
            row = await repo.get(identity.conversation_id, identity.turn_id)
            if row is not None and row.status == TurnStatus.RUNNING:
                row.status = TurnStatus.FAILED
                row.failure_reason = reason.value
                elapsed = await repo.current_time() - row.created_at
                row.latency_ms = min(2_147_483_647, max(0, int(elapsed.total_seconds() * 1000)))
                logger.info("turn_failed", turn_id=str(identity.turn_id), reason=reason.value)
        if row is not None:
            record_turn_status(row.status.value)

    async def reconcile(self) -> None:
        """Finalize only abandoned pre-startup turns, without checkpoint replay."""
        async with self.database.session() as session:
            cutoff = await lifecycle.database_time(session)
            users = await lifecycle.user_ids(session)
        for user_id in users:
            async with self.database.session() as session:
                candidates = await lifecycle.running_turns(session, user_id, cutoff)
            for candidate in candidates:
                await self._reconcile_candidate(candidate)

    async def _reconcile_candidate(self, candidate: lifecycle.RunningIdentity) -> None:
        async with self.lease(candidate.conversation_id) as acquired:
            if acquired:
                await self._interrupt(candidate)

    async def _interrupt(self, candidate: lifecycle.RunningIdentity) -> None:
        async with (
            asyncio.timeout(self.timeout_s),
            self.database.session() as session,
            session.begin(),
        ):
            if not await lifecycle.try_graph_guard(session, candidate.conversation_id):
                return
            repo = TurnRepository(session, candidate.user_id)
            await repo.lock_conversation(candidate.conversation_id)
            row = await repo.get(candidate.conversation_id, candidate.turn_id)
            if row is not None and row.status == TurnStatus.RUNNING:
                row.status = TurnStatus.FAILED
                row.failure_reason = FailureKind.INTERRUPTED.value
                elapsed = await repo.current_time() - row.created_at
                row.latency_ms = min(2_147_483_647, max(0, int(elapsed.total_seconds() * 1000)))
                logger.info("turn_interrupted", turn_id=str(row.id))
