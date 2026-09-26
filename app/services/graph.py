"""Lifespan-owned PostgreSQL saver and ownership-checked internal graph invocation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

import psycopg
import structlog
from langgraph.types import Command, Overwrite
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

    from app.agents.runtime import RuntimeContext
    from app.core.config_models import DatabaseSettings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents import contracts, failures, state
from app.agents.data import state as data_state
from app.agents.graph import RECURSION_LIMIT, PhaseOneGraph, build
from app.agents.knowledge import state as knowledge_state
from app.agents.state import GRAPH_VERSION, AgentState, GraphInput, GraphOutput
from app.core.errors import CheckpointError, ConflictError, DeadlineExceededError
from app.retrieval import config as retrieval_config
from app.schemas import (
    clarification,
    corpus,
    ingestion,
    knowledge,
    knowledge_query,
    mcp,
    memory,
    memory_retrieval,
    metric_resolution,
    metrics,
    model_runtime,
    retrieval,
    sanity,
    schema_catalog,
    sql_correction,
    synthesis,
)
from app.services import periods
from app.services.deadline_finalization import finalize_deadline

logger = structlog.get_logger(__name__)


def checkpoint_conninfo(settings: DatabaseSettings) -> str:
    """psycopg keyword escaping handles reserved characters without URL confusion."""
    return make_conninfo(
        host=settings.host,
        port=settings.port,
        dbname=settings.app_db,
        user=settings.app_user,
        password=settings.app_password.get_secret_value(),
    )


def serializer() -> JsonPlusSerializer:
    """Only our durable contract types may be reconstructed; never runtime objects."""
    allowed = [
        (module.__name__, name)
        for module in (
            contracts,
            failures,
            state,
            data_state,
            knowledge_state,
            retrieval_config,
            clarification,
            corpus,
            ingestion,
            knowledge,
            knowledge_query,
            model_runtime,
            retrieval,
            mcp,
            memory,
            memory_retrieval,
            metric_resolution,
            metrics,
            sanity,
            schema_catalog,
            sql_correction,
            synthesis,
            periods,
        )
        for name, value in vars(module).items()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    # Historical checkpoints encoded the enum before it moved to the shared schema.
    allowed.append((contracts.__name__, "SanityFlag"))
    # Step 3 knowledge checkpoints used the specialist-local terminology class.
    allowed.append((knowledge_state.__name__, "TerminologyProjection"))
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=allowed)


class GraphService:
    """Own the independent async psycopg pool, never any business DB credential."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        # Admission slots reserve at least one pool connection for checkpoint writes.
        self._guard_slots = asyncio.Semaphore(max(1, settings.pool_size - 1))
        self.pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, object]]] | None = None
        self.graph: PhaseOneGraph | None = None

    async def start(self) -> None:
        """Verify operator-installed checkpoint schema without issuing DDL."""
        if self.pool is not None:
            return
        pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, object]]] = AsyncConnectionPool(
            checkpoint_conninfo(self.settings),
            min_size=0,
            max_size=max(2, self.settings.pool_size),
            timeout=self.settings.pool_timeout_s,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,
                "row_factory": dict_row,
                "connect_timeout": 5,
                "options": f"-c statement_timeout={int(self.settings.command_timeout_s * 1000)}",
            },
        )
        self.pool = pool
        try:
            async with asyncio.timeout(
                self.settings.connect_timeout_s + self.settings.command_timeout_s
            ):
                await pool.open()
                await self._verify(pool)
                self.graph = build(AsyncPostgresSaver(pool, serde=serializer()))
        except (psycopg.Error, TimeoutError, CheckpointError) as exc:
            await self.aclose()
            raise CheckpointError() from exc
        logger.info("graph_started", graph_version=GRAPH_VERSION)

    async def _verify(
        self, pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, object]]]
    ) -> None:
        async with pool.connection() as connection:
            cursor = await connection.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            versions = [row["v"] for row in await cursor.fetchall()]
            if versions != list(range(len(AsyncPostgresSaver.MIGRATIONS))):
                raise CheckpointError()
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                result = await connection.execute("SELECT to_regclass(%s) AS relation", (table,))
                row = await result.fetchone()
                if row is None or row["relation"] is None:
                    raise CheckpointError()

    async def aclose(self) -> None:
        """Idempotently release the graph pool, including failed startup."""
        pool, self.pool, self.graph = self.pool, None, None
        if pool is not None:
            await pool.close(timeout=self.settings.connect_timeout_s)

    @asynccontextmanager
    async def _guard(self, ctx: RuntimeContext) -> AsyncIterator[None]:
        if self.pool is None:
            raise CheckpointError()
        async with self._guard_slots, self.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
                (str(ctx.identity.conversation_id),),
            )
            row = await cursor.fetchone()
            if row is None or not row["acquired"]:
                raise ConflictError("conversation graph is already executing")
            try:
                await ctx.conversations.prepare(ctx.identity)
                yield
            finally:
                try:
                    await connection.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                        (str(ctx.identity.conversation_id),),
                    )
                except (psycopg.Error, asyncio.CancelledError):
                    # A session lock must never survive a failed release in a pooled connection.
                    await connection.close()
                    raise

    async def invoke(
        self,
        ctx: RuntimeContext,
        *,
        resume: bool = False,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> GraphOutput:
        """Fresh turns must be new; explicit recovery validates owned/versioned state."""
        if self.graph is None:
            raise CheckpointError()
        ctx.deadline.check("graph")
        config: RunnableConfig = {
            "configurable": {"thread_id": str(ctx.identity.turn_id)},
            "recursion_limit": RECURSION_LIMIT,
            "callbacks": callbacks or [],
        }
        try:
            async with asyncio.timeout(ctx.finalization_deadline.remaining()), self._guard(ctx):
                result = await self._run(ctx, config, resume)
                logger.info(
                    "graph_completed",
                    turn_id=str(ctx.identity.turn_id),
                    conversation_id=str(ctx.identity.conversation_id),
                    user_id=str(ctx.identity.user_id),
                    request_id=ctx.trace_id,
                    status=result.status,
                )
                return result
        except TimeoutError as exc:
            raise DeadlineExceededError() from exc
        except psycopg.Error as exc:
            raise CheckpointError() from exc

    async def _run(self, ctx: RuntimeContext, config: RunnableConfig, resume: bool) -> GraphOutput:
        if self.graph is None:
            raise CheckpointError()
        timer = asyncio.timeout(ctx.deadline.remaining())
        try:
            async with timer:
                graph_input = await self._input(ctx, config, resume)
                output = await self.graph.ainvoke(
                    graph_input, config, context=ctx, durability="sync"
                )
                result = GraphOutput.model_validate(output)
        except TimeoutError:
            if not timer.expired():
                raise
        else:
            if ctx.deadline.remaining() > 0:
                return result
        # The invocation has unwound its tasks and pending checkpoint writes before
        # this read. Latest-state reads include completed parent-node pending writes.
        checkpoint = await self.graph.aget_state(config)
        if not checkpoint.values:
            raise DeadlineExceededError()
        prior = self._owned_state(checkpoint.values, ctx)
        return await finalize_deadline(prior, ctx)

    @staticmethod
    def _owned_state(values: object, ctx: RuntimeContext) -> AgentState:
        try:
            prior = AgentState.model_validate(values)
        except ValidationError as exc:
            raise ConflictError("unsupported checkpoint contract") from exc
        if (
            prior.graph_version != GRAPH_VERSION
            or prior.user_id != ctx.identity.user_id
            or prior.conversation_id != ctx.identity.conversation_id
            or prior.turn_id != ctx.identity.turn_id
        ):
            raise ConflictError("checkpoint identity/version mismatch")
        return prior

    async def _input(
        self, ctx: RuntimeContext, config: RunnableConfig, resume: bool
    ) -> GraphInput | Command[Literal["synthesize", "format_answer", "persist_evidence"]] | None:
        if self.graph is None:
            raise CheckpointError()
        checkpoint = await self.graph.aget_state(config)
        if not checkpoint.values:
            if resume:
                raise ConflictError("checkpoint not found")
            return GraphInput(**ctx.identity.model_dump())
        if checkpoint.values.get("graph_version") != GRAPH_VERSION:
            raise ConflictError("checkpoint graph version mismatch")
        try:
            prior = AgentState.model_validate(checkpoint.values)
        except ValidationError as exc:
            raise ConflictError("unsupported checkpoint contract") from exc
        if (
            prior.graph_version != GRAPH_VERSION
            or prior.user_id != ctx.identity.user_id
            or prior.conversation_id != ctx.identity.conversation_id
            or prior.turn_id != ctx.identity.turn_id
        ):
            raise ConflictError("checkpoint identity/version mismatch")
        if not resume:
            raise ConflictError("turn already has a checkpoint")
        if checkpoint.next or prior.answer is not None or prior.clarification is not None:
            return None
        return await self._recovery_input(ctx, prior)

    async def _recovery_input(
        self, ctx: RuntimeContext, prior: AgentState
    ) -> Command[Literal["synthesize", "format_answer", "persist_evidence"]]:
        """Recover only from the owned immutable bundle, including a lost commit receipt."""
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.data is None and bundle.knowledge is None:
            raise ConflictError("failed turn has no committed evidence to recover")
        if prior.evidence_refs is None or (
            prior.evidence_refs.data_snapshot_id is None
            and prior.evidence_refs.knowledge_snapshot_id is None
        ):
            # A commit can succeed before its node result reaches the checkpointer.
            # Replay the persistence barrier using only the verified committed bundle.
            return Command(
                update={
                    "failures": Overwrite([]),
                    "data_evidence": bundle.data.data if bundle.data else None,
                    "knowledge_evidence": bundle.knowledge.knowledge if bundle.knowledge else None,
                },
                goto="persist_evidence",
            )
        if prior.evidence_refs != bundle.refs:
            raise ConflictError("checkpoint references differ from committed evidence")
        # Recovery input writes overlay the prior checkpoint's get_state() view.
        # Historical auditing reads the saver's original channel_values instead.
        return Command(
            update={"failures": Overwrite([])},
            goto="synthesize"
            if prior.route and prior.route.route is contracts.Route.BOTH
            else "format_answer",
        )
