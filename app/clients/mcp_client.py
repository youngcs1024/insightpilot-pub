"""Long-lived MCP v2 client with one retry owner and a concurrency-safe breaker."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import Literal

import anyio
import httpx2
import structlog
from asgi_correlation_id import correlation_id
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, CallToolResult
from pydantic import ValidationError

from app.core.background import spawn
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.core.errors import (
    AuthenticationError,
    InsightPilotError,
    McpPolicyRejected,
    McpResultError,
    McpUnavailableError,
    SchemaDriftError,
    SchemaMetadataError,
    SqlExecutionError,
    SqlTimeoutError,
    UpstreamUnavailableError,
)
from app.core.observability import TraceMetadata, observe
from app.core.retry import run_operation
from app.schemas.mcp import (
    Contract,
    McpErrorCode,
    McpErrorPayload,
    QueryArguments,
    QueryResultPayload,
    ValidationStatus,
)
from app.schemas.metric_tools import MetricFragment, ResolveMetricArgs
from app.schemas.schema_tools import GetSchemaArgs, SchemaResponse, SchemaToolError

logger = structlog.get_logger(__name__)
BREAKER_FAILURES = 5
BREAKER_RECOVERY_S = 30
_CORRELATION_META = "insightpilot/request_id"

type SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]


async def attach_correlation_header(request: httpx2.Request) -> None:
    """Copy the current call's metadata into its own HTTP POST, without shared headers."""
    if request.method != "POST":
        return
    try:
        payload = json.loads(request.content)
    except (ValueError, httpx2.RequestNotRead):
        return
    if payload.get("method") != "tools/call":
        return
    value = payload.get("params", {}).get("_meta", {}).get(_CORRELATION_META)
    if isinstance(value, str):
        request.headers["X-Request-ID"] = value


class CircuitBreaker:
    """Count final logical failures, with a single admitted recovery probe."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self.probing = False

    def enter(self) -> bool:
        """Return whether this caller owns the half-open probe; never await here."""
        if self.opened_at is None:
            return False
        if self.probing or self.clock() - self.opened_at < BREAKER_RECOVERY_S:
            raise McpUnavailableError()
        self.probing = True
        return True

    def success(self) -> None:
        """A successful operation proves the dependency recovered."""
        self.failures = 0
        self.opened_at = None
        self.probing = False

    def failed(self) -> None:
        """Only callers with a final retryable upstream failure may call this."""
        self.failures += 1
        if self.failures >= BREAKER_FAILURES:
            self.opened_at = self.clock()
        self.probing = False

    def release(self, probe: bool) -> None:
        """Policy errors/cancellation never increment failures or close the breaker."""
        if probe:
            self.probing = False


def decode_result(result: CallToolResult) -> QueryResultPayload:
    """Interpret structured data exclusively; human-readable content is not control flow."""
    try:
        if not result.is_error:
            return QueryResultPayload.model_validate(result.structured_content)
        failure = McpErrorPayload.model_validate(result.structured_content)
    except ValidationError as exc:
        raise McpResultError() from exc
    match failure.code:
        case McpErrorCode.POLICY_REJECTED:
            raise McpPolicyRejected(
                failure.status or ValidationStatus.UNSAFE,
                failure.reasons,
                column_name=failure.column_name,
            )
        case McpErrorCode.SQL_TIMEOUT:
            raise SqlTimeoutError()
        case McpErrorCode.SQL_EXECUTION_FAILED:
            raise SqlExecutionError(failure.sql_error)
        case McpErrorCode.UNAVAILABLE:
            raise McpUnavailableError()
        case McpErrorCode.INVALID_RESULT:
            raise McpResultError()


def raise_metric_error(result: CallToolResult) -> None:
    """Decode schema-health and policy failures without inspecting message text."""
    try:
        failure = SchemaToolError.model_validate(result.structured_content)
    except ValidationError:
        decode_result(result)
        raise McpResultError() from None
    if failure.code == "SCHEMA_DRIFT":
        raise SchemaDriftError(report=failure.report)
    raise SchemaMetadataError()


def transport_failure(exc: BaseException) -> InsightPilotError:
    """Unwrap transport task groups using exception types/status, never messages."""
    if isinstance(exc, BaseExceptionGroup):
        mapped = [transport_failure(item) for item in exc.exceptions]
        return next((item for item in mapped if not item.retryable), mapped[0])
    if isinstance(exc, InsightPilotError):
        return exc
    if (isinstance(exc, MCPError) and exc.code in (CONNECTION_CLOSED, REQUEST_TIMEOUT)) or (
        isinstance(exc, httpx2.HTTPStatusError)
        and exc.response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        return McpUnavailableError()
    if isinstance(exc, httpx2.HTTPStatusError):
        return AuthenticationError() if exc.response.status_code in (401, 403) else McpResultError()
    if isinstance(
        exc,
        (
            httpx2.TransportError,
            OSError,
            TimeoutError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
        ),
    ):
        return McpUnavailableError()
    return McpResultError()


class McpClient:
    """A dedicated owner task enters/exits SDK cancel scopes in the same task."""

    def __init__(
        self, settings: MCPSettings, *, session_factory: SessionFactory | None = None
    ) -> None:
        self.settings = settings
        self._factory = session_factory or self._session_context
        self._lock = asyncio.Lock()
        self._session: ClientSession | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._closed = False
        self.breaker = CircuitBreaker()

    @asynccontextmanager
    async def _session_context(self) -> AsyncIterator[ClientSession]:
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": "Bearer " + self.settings.auth_token.get_secret_value()},
                event_hooks={"request": [attach_correlation_header]},
                timeout=self.settings.timeout_s,
                trust_env=False,
            ) as http,
            streamable_http_client(str(self.settings.base_url), http_client=http) as (read, write),
            ClientSession(read, write, read_timeout_seconds=self.settings.timeout_s) as session,
        ):
            await session.initialize()
            yield session

    async def _run_session(self, ready: asyncio.Future[ClientSession], stop: asyncio.Event) -> None:
        session: ClientSession | None = None
        try:
            async with self._factory() as session:
                self._session = session
                if not ready.done():
                    ready.set_result(session)
                await stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(transport_failure(exc))
            logger.exception("mcp_session_failed")
        finally:
            if self._session is session:
                self._session = None
            if not ready.done():
                ready.cancel()

    async def connect(self) -> None:
        """Establish one bounded connection; startup performs no retry loop."""
        try:
            async with asyncio.timeout(self.settings.timeout_s):
                await self._get_session()
        except TimeoutError as exc:
            raise McpUnavailableError() from exc

    async def _get_session(self) -> ClientSession:
        async with self._lock:
            if self._closed:
                raise McpUnavailableError()
            if self._session is not None:
                return self._session
            if self._owner is not None and not self._owner.done():
                await self._stop_owner()
            self._stop = asyncio.Event()
            ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
            self._owner = spawn(self._run_session(ready, self._stop), name="mcp-session-owner")
            try:
                return await ready
            except BaseException:
                self._stop.set()
                self._owner.cancel()
                raise

    async def _stop_owner(self) -> None:
        self._stop.set()
        if self._owner is None:
            return
        done, _ = await asyncio.wait({self._owner}, timeout=2)
        if not done:
            self._owner.cancel()
            done, _ = await asyncio.wait({self._owner}, timeout=2)
            if not done:
                raise McpUnavailableError()
        self._session = None

    async def aclose(self) -> None:
        """Reject new calls and close the long-lived session with a bounded wait."""
        async with self._lock:
            self._closed = True
            await self._stop_owner()

    async def call_tool(
        self,
        name: Literal["execute_readonly_query"],
        args: QueryArguments,
        *,
        deadline: Deadline,
    ) -> QueryResultPayload:
        """Call the single supported tool; no business-database fallback exists."""
        if name != "execute_readonly_query":
            raise McpResultError()
        return await self._with_retry(lambda: self._call(name, args), deadline)

    async def get_schema(self, args: GetSchemaArgs, *, deadline: Deadline) -> SchemaResponse:
        """Read server-rendered metadata through the existing session and retry owner."""
        return await self._with_retry(lambda: self._schema_call(args), deadline)

    async def resolve_metric(
        self, args: ResolveMetricArgs, *, deadline: Deadline
    ) -> MetricFragment:
        """Validate a rendered metric through the existing bounded MCP session."""
        return await self._with_retry(lambda: self._metric_call(args), deadline)

    async def _metric_call(self, args: ResolveMetricArgs) -> MetricFragment:
        with observe("mcp_metric", TraceMetadata(tool="resolve_metric")):
            result = await self._raw_call("resolve_metric", args)
            if result.is_error:
                raise_metric_error(result)
            try:
                return MetricFragment.model_validate(result.structured_content)
            except ValidationError as exc:
                raise McpResultError() from exc

    async def _schema_call(self, args: GetSchemaArgs) -> SchemaResponse:
        with observe("mcp_schema", TraceMetadata(tool="get_schema")):
            result = await self._raw_call("get_schema", args)
            if result.is_error:
                try:
                    failure = SchemaToolError.model_validate(result.structured_content)
                except ValidationError:
                    decode_result(result)
                    raise McpResultError() from None
                if failure.code == "SCHEMA_DRIFT":
                    raise SchemaDriftError(report=failure.report)
                raise SchemaMetadataError()
            try:
                return SchemaResponse.model_validate(result.structured_content)
            except ValidationError as exc:
                raise McpResultError() from exc

    async def _with_retry[ResultT](
        self, operation: Callable[[], Awaitable[ResultT]], deadline: Deadline
    ) -> ResultT:
        deadline.check("mcp_call")
        probe = self.breaker.enter()
        try:
            result = await run_operation(
                operation,
                deadline=deadline,
                timeout_s=self.settings.timeout_s,
                name="mcp_call",
                attempts=2,
            )
        except UpstreamUnavailableError as exc:
            if exc.retryable:
                self.breaker.failed()
                raise McpUnavailableError() from exc
            raise
        else:
            self.breaker.success()
            return result
        finally:
            self.breaker.release(probe)

    async def _call(self, name: str, args: QueryArguments) -> QueryResultPayload:
        with observe("mcp_execute", TraceMetadata(tool=name)) as observation:
            result = await self._execute_call(name, args)
            if observation is not None:
                observation.update(
                    TraceMetadata(
                        row_count=result.row_count,
                        execution_ms=result.execution_ms,
                        mcp_call_id=result.mcp_call_id,
                    )
                )
            return result

    async def _execute_call(self, name: str, args: QueryArguments) -> QueryResultPayload:
        return decode_result(await self._raw_call(name, args))

    async def _raw_call(self, name: str, args: Contract) -> CallToolResult:
        try:
            session = await self._get_session()
            request_id = correlation_id.get()
            if request_id is None:
                result = await session.call_tool(name, arguments=args.model_dump(mode="json"))
            else:
                result = await session.call_tool(
                    name,
                    arguments=args.model_dump(mode="json"),
                    meta={_CORRELATION_META: request_id},
                )
        except asyncio.CancelledError:
            # End the session after cancellation; late responses cannot poison its reuse.
            self._stop.set()
            self._session = None
            raise
        except Exception as exc:
            self._stop.set()
            self._session = None
            raise transport_failure(exc) from exc
        if not isinstance(result, CallToolResult):
            raise McpResultError()
        return result
