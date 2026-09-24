"""Application factory, separate from the fail-fast process configuration import."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from app import __version__
from app.api.exception_handlers import (
    handle_http,
    handle_known,
    handle_request_validation,
    handle_unknown,
)
from app.api.v1.health import router as health_router
from app.api.v1.router import router as api_router
from app.clients.mcp_client import McpClient
from app.clients.model_resilience import STARTUP_BUDGET_S
from app.clients.model_runtime import ModelRuntimeClient, ModelRuntimeProbe
from app.core.background import shutdown as shutdown_background
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError, McpResultError, McpToolSchemaError
from app.core.limiter import AuthLimiter
from app.core.logging import setup_logging
from app.core.middleware import DeadlineMiddleware, LoggingContextMiddleware, RequestIdMiddleware
from app.core.observability import Observability
from app.db.session import Database
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.search_store import HybridSearchStore
from app.services.auth import AuthService
from app.services.chat import ChatService
from app.services.memory.extract import MemoryExtractionService
from app.services.clarification_capabilities import ClarificationCapabilityService
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from app.services.health import HealthService, MCPProbe, PostgreSQLProbe
from app.services.knowledge_generation import KnowledgeGenerationService
from app.services.llm.service import LlmService
from app.services.metrics import MetricService
from app.services.schema_catalog import SchemaCatalogService
from app.services.schema_tokens import SchemaTokenCounter
from model_runtime.errors import ModelError

logger = structlog.get_logger(__name__)


async def _close_database(database: Database) -> None:
    try:
        await database.aclose()
    except Exception:
        logger.exception("database_cleanup_failed")


async def _close_mcp(client: McpClient) -> None:
    try:
        await client.aclose()
    except Exception:
        logger.exception("mcp_cleanup_failed")


async def _close_llm(client: LlmService) -> None:
    try:
        await client.aclose()
    except Exception:
        logger.exception("llm_cleanup_failed")


async def _close_graph(graph: GraphService) -> None:
    try:
        await graph.aclose()
    except Exception:
        logger.exception("graph_cleanup_failed")


async def _close_model(client: ModelRuntimeClient | None) -> None:
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:
        logger.exception("model_cleanup_failed")


async def _close_retrieval(store: HybridSearchStore | None) -> None:
    if store is not None:
        try:
            await store.aclose()
        except Exception:
            logger.exception("retrieval_cleanup_failed")


async def _close_resources(  # noqa: PLR0913, PLR0917 -- one explicit lifecycle owner per resource.
    service: HealthService,
    database: Database,
    client: McpClient,
    llm: LlmService,
    graph: GraphService,
    timeout_s: float,
    model: ModelRuntimeClient | None,
    store: HybridSearchStore | None = None,
) -> None:
    try:
        async with asyncio.timeout(timeout_s), asyncio.TaskGroup() as group:
            group.create_task(shutdown_background(timeout_s))
            group.create_task(service.aclose())
            group.create_task(_close_database(database))
            group.create_task(_close_mcp(client))
            group.create_task(_close_llm(llm))
            group.create_task(_close_graph(graph))
            group.create_task(_close_model(model))
            group.create_task(_close_retrieval(store))
    except TimeoutError:
        logger.exception("application_shutdown_timeout")


def create_app(  # noqa: PLR0913, PLR0915 -- explicit resources and middleware composition.
    settings: Settings,
    *,
    health_service: HealthService | None = None,
    database: Database | None = None,
    mcp_client: McpClient | None = None,
    llm_service: LlmService | None = None,
    graph_service: GraphService | None = None,
    observability: Observability | None = None,
    model_runtime_client: ModelRuntimeClient | None = None,
) -> FastAPI:
    """Create an isolated application with typed, replaceable readiness dependencies."""
    observability = observability or Observability(settings)
    llm_service = llm_service or LlmService(settings.llm)
    database = database or Database(settings.database)
    mcp_client = mcp_client or McpClient(settings.mcp)
    graph_service = graph_service or GraphService(settings.database)
    if settings.retrieval.enabled and settings.model_runtime is not None:
        model_runtime_client = model_runtime_client or ModelRuntimeClient(settings.model_runtime)
    service = health_service or HealthService(
        PostgreSQLProbe(database),
        MCPProbe(mcp_client),
        settings.health,
        model=ModelRuntimeProbe(model_runtime_client)
        if settings.retrieval.enabled and model_runtime_client is not None
        else None,
        mcp_tools_ready=lambda: mcp_client.tools_loaded,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        startup_deadline = Deadline(time.monotonic() + STARTUP_BUDGET_S)
        setup_logging(settings)
        app.state.ready = False
        store = None
        try:
            await observability.start()
            await llm_service.start()
            database.start()
            if settings.retrieval.enabled:
                store = HybridSearchStore(settings.retrieval.milvus)
                app.state.retrieval = RetrievalPipeline(
                    database, store, model_runtime_client, settings.retrieval
                )
            await graph_service.start()
            await app.state.metrics.validate_startup()
            await app.state.chat.reconcile()
            try:
                await mcp_client.connect()
                await mcp_client.refresh_tools(deadline=startup_deadline)
            except (McpToolSchemaError, McpResultError):
                logger.exception("mcp_tool_schema_invalid")
                raise
            except Exception:
                logger.exception("mcp_startup_unavailable")
            if settings.retrieval.enabled and model_runtime_client is not None:
                try:
                    await model_runtime_client.warmup(deadline=startup_deadline)
                except ModelError as exc:
                    logger.exception("model_startup_unavailable", code=exc.code)
            checks = await service.start()
            app.state.ready = checks.ready
            logger.info("application_started", ready=checks.ready, version=__version__)
            yield
        finally:
            app.state.ready = False
            app.state.schema_catalog.clear()
            service.active = False
            await _close_resources(
                service,
                database,
                mcp_client,
                llm_service,
                graph_service,
                settings.http.shutdown_timeout_s,
                model_runtime_client,
                store,
            )
            await observability.aclose()
            logger.info("application_stopped")

    app = FastAPI(title="InsightPilot", version=__version__, lifespan=lifespan)
    _install_chat(app, settings, database, graph_service, llm_service)
    app.state.observability = observability
    app.state.chat.observability = observability
    app.state.graph = graph_service
    app.state.evidence = EvidenceService(database)
    app.state.conversations = ConversationService(database)
    app.state.llm = llm_service
    app.state.mcp = mcp_client
    app.state.model_runtime = model_runtime_client
    app.state.retrieval = None
    app.state.knowledge_generation = KnowledgeGenerationService(llm_service)
    app.state.metrics = MetricService(database, settings.database)
    app.state.clarification_capabilities = ClarificationCapabilityService(
        database, settings.database
    )
    app.state.schema_token_counter = SchemaTokenCounter()
    app.state.schema_catalog = SchemaCatalogService(database, mcp_client, settings.schema_catalog)
    app.state.auth = AuthService(database, settings.security)
    app.state.auth_limiter = AuthLimiter(settings.rate_limits)
    app.state.health = service
    app.state.database = database
    app.state.ready = False
    app.add_exception_handler(InsightPilotError, handle_known)
    app.add_exception_handler(RequestValidationError, handle_request_validation)
    app.add_exception_handler(HTTPException, handle_http)
    app.add_exception_handler(Exception, handle_unknown)
    # Starlette prepends each layer: request ID must be added last, outside CORS.
    app.add_middleware(
        DeadlineMiddleware,
        timeout_s=settings.http.request_timeout_s,
        finalization_grace_s=settings.http.finalization_grace_s,
    )
    app.add_middleware(LoggingContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.http.cors_origins,
        allow_credentials=settings.http.cors_allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Idempotency-Replayed", "Retry-After"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health_router)
    app.include_router(api_router, prefix="/api/v1")
    return app


def _install_chat(
    app: FastAPI, settings: Settings, database: Database, graph: GraphService, llm: LlmService
) -> None:
    app.state.settings = settings
    app.state.chat = ChatService(
        database, settings, graph, memory=MemoryExtractionService(database, settings, llm)
    )
