"""Long-lived MCP v2 client with one retry owner and a concurrency-safe breaker."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal, cast

import anyio
import httpx2
import structlog
from asgi_correlation_id import correlation_id
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CONNECTION_CLOSED,
    REQUEST_TIMEOUT,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    RequestParamsMeta,
    TextContent,
    Tool,
)
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from app.clients.mcp_langchain import mcp_tool_to_langchain
from app.core.background import spawn
from app.core.errors import (
    InsightPilotError,
    McpAuthenticationError,
    McpCallTimeoutError,
    McpPolicyRejected,
    McpResultError,
    McpUnavailableError,
    OperationTimeoutError,
    SchemaDriftError,
    SchemaMetadataError,
    SqlExecutionError,
    SqlTimeoutError,
    UpstreamUnavailableError,
)
from app.core.observability import TraceMetadata, observe
from app.core.retry import is_retryable, run_operation
from app.schemas.mcp import (
    DiscoveredToolResult,
    McpErrorCode,
    McpErrorPayload,
    McpReadyResponse,
    QueryArguments,
    QueryResultPayload,
    ValidationStatus,
)
from app.schemas.metric_tools import MetricFragment, ResolveMetricArgs
from app.schemas.schema_tools import GetSchemaArgs, SchemaResponse, SchemaToolError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from app.core.config_models import MCPSettings
    from app.core.deadline import Deadline

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


async def preserve_http_failure(response: httpx2.Response) -> None:
    """Keep HTTP status available before the SDK turns it into a generic JSON-RPC error."""
    status = response.status_code
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or (
        response.request.method == "POST"
        and (
            status in (HTTPStatus.NOT_FOUND, HTTPStatus.REQUEST_TIMEOUT)
            or status >= HTTPStatus.INTERNAL_SERVER_ERROR
        )
    ):
        response.raise_for_status()


def retry_mcp_failure(exc: BaseException) -> bool:
    """Repeat only failures known to be safe; a call timeout may still be running."""
    return not isinstance(exc, OperationTimeoutError) and is_retryable(exc)


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
        """Count a final availability failure once, including a non-retryable timeout."""
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


def _http_failure(exc: httpx2.HTTPStatusError) -> InsightPilotError:
    status = exc.response.status_code
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return McpAuthenticationError()
    if status in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.GATEWAY_TIMEOUT):
        return McpCallTimeoutError()
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR or (
        status == HTTPStatus.NOT_FOUND and "mcp-session-id" in exc.request.headers
    ):
        return McpUnavailableError()
    return McpResultError()


def _sdk_failure(exc: MCPError) -> InsightPilotError:
    if exc.code == REQUEST_TIMEOUT:
        return McpCallTimeoutError()
    return McpUnavailableError() if exc.code == CONNECTION_CLOSED else McpResultError()


def _socket_failure(exc: BaseException) -> InsightPilotError:
    if isinstance(exc, (httpx2.TimeoutException, TimeoutError)):
        return McpCallTimeoutError()
    return McpUnavailableError()


def transport_failure(exc: BaseException) -> InsightPilotError:
    """Unwrap transport task groups using exception types/status, never messages."""
    if isinstance(exc, BaseExceptionGroup):
        mapped = [transport_failure(item) for item in exc.exceptions]
        return next((item for item in mapped if not item.retryable), mapped[0])
    if isinstance(exc, InsightPilotError):
        return exc
    if isinstance(exc, MCPError):
        return _sdk_failure(exc)
    if isinstance(exc, httpx2.HTTPStatusError):
        return _http_failure(exc)
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
        return _socket_failure(exc)
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
        self._refresh_lock = asyncio.Lock()
        self._tools: tuple[BaseTool, ...] = ()
        self._descriptors: dict[str, Tool] = {}
        self._argument_models: dict[str, type[BaseModel]] = {}
        self._tools_loaded = False

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return an immutable snapshot of the successfully built tool cache."""
        return self._tools

    @property
    def tools_loaded(self) -> bool:
        """Distinguish a completed discovery from startup without MCP access."""
        return self._tools_loaded

    @asynccontextmanager
    async def _session_context(self) -> AsyncIterator[ClientSession]:
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": "Bearer " + self.settings.auth_token.get_secret_value()},
                event_hooks={
                    "request": [attach_correlation_header],
                    "response": [preserve_http_failure],
                },
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
            raise McpCallTimeoutError() from exc

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

    async def list_tools(self, *, deadline: Deadline) -> tuple[Tool, ...]:
        """Discover every descriptor through the shared authenticated session."""
        return await self._with_retry(self._list_tools, deadline)

    async def _list_tools(self) -> tuple[Tool, ...]:
        try:
            session = await self._get_session()
            return await self._list_descriptors(session)
        except asyncio.CancelledError:
            self._stop.set()
            self._session = None
            raise
        except Exception as exc:
            self._stop.set()
            self._session = None
            raise transport_failure(exc) from exc

    @staticmethod
    async def _list_descriptors(session: ClientSession) -> tuple[Tool, ...]:
        tools: list[Tool] = []
        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor is not None else None
            response = await session.list_tools(params=params)
            if not isinstance(response, ListToolsResult):
                raise McpResultError()
            tools.extend(response.tools)
            cursor = response.next_cursor
            if cursor is None:
                return tuple(tools)
            if cursor in seen_cursors:
                raise McpResultError()
            seen_cursors.add(cursor)

    async def refresh_tools(self, *, deadline: Deadline) -> tuple[BaseTool, ...]:
        """Build a full replacement cache before making descriptors available."""
        async with self._refresh_lock:
            descriptors = await self.list_tools(deadline=deadline)
            return self._install_descriptors(descriptors)

    def _install_descriptors(self, descriptors: tuple[Tool, ...]) -> tuple[BaseTool, ...]:
        by_name = {tool.name: tool for tool in descriptors}
        if len(by_name) != len(descriptors):
            raise McpResultError()
        built = tuple(mcp_tool_to_langchain(tool, self) for tool in descriptors)
        for name in sorted(set(self._descriptors) | set(by_name)):
            if self._descriptors.get(name) != by_name.get(name) and self._tools_loaded:
                logger.warning("mcp_tool_descriptor_changed", tool=name)
        self._descriptors = by_name
        self._tools = built
        self._argument_models = {
            tool.name: cast("type[BaseModel]", tool.args_schema) for tool in built
        }
        self._tools_loaded = True
        logger.info("mcp_tools_loaded", count=len(built))
        return built

    async def health(self) -> None:
        """Check physical readiness and auth without retries or breaker mutation."""
        ready_url = str(self.settings.base_url).removesuffix("/mcp") + "/ready"
        try:
            async with httpx2.AsyncClient(timeout=self.settings.timeout_s, trust_env=False) as http:
                response = await http.get(ready_url)
                response.raise_for_status()
                try:
                    ready = McpReadyResponse.model_validate(response.json())
                except (ValidationError, ValueError) as exc:
                    raise McpResultError() from exc
                if not ready.ready:
                    raise McpUnavailableError()
            async with self._factory() as session:
                descriptors = await self._list_descriptors(session)
            async with self._refresh_lock:
                if (
                    not self._tools_loaded
                    or {tool.name: tool for tool in descriptors} != self._descriptors
                ):
                    self._install_descriptors(descriptors)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise transport_failure(exc) from exc

    async def invoke_discovered_tool(
        self, name: str, args: BaseModel, *, deadline: Deadline
    ) -> DiscoveredToolResult:
        """Execute one cached descriptor with a typed Pydantic input model."""
        model = self._argument_models.get(name)
        if model is None or not isinstance(args, model):
            raise McpResultError()
        return await self._with_retry(lambda: self._discovered_call(name, args), deadline)

    async def _discovered_call(self, name: str, args: BaseModel) -> DiscoveredToolResult:
        with observe("mcp_discovered_tool", TraceMetadata(tool=name)):
            result = await self._raw_call(name, args, omit_unset=True)
            if result.is_error:
                raise_metric_error(result)
            payload = result.structured_content
            if payload is not None:
                return DiscoveredToolResult(text=self._render_structured(name, payload))
            if name in {"execute_readonly_query", "get_schema", "resolve_metric"}:
                raise McpResultError()
            texts = [item.text for item in result.content if isinstance(item, TextContent)]
            if not texts or len(texts) != len(result.content):
                raise McpResultError()
            text = "\n".join(texts)
            if not text:
                raise McpResultError()
            return DiscoveredToolResult(text=text)

    @staticmethod
    def _render_structured(name: str, payload: object) -> str:
        try:
            if name == "execute_readonly_query":
                return QueryResultPayload.model_validate(payload).model_dump_json()
            if name == "get_schema":
                return SchemaResponse.model_validate(payload).model_dump_json()
            if name == "resolve_metric":
                return MetricFragment.model_validate(payload).model_dump_json()
            return json.dumps(TypeAdapter(JsonValue).validate_python(payload), allow_nan=False)
        except (ValidationError, TypeError, ValueError) as exc:
            raise McpResultError() from exc

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
                attempts=3,
                retry_policy=retry_mcp_failure,
            )
        except McpAuthenticationError:
            logger.exception("mcp_authentication_failed", exc_info=False)
            raise
        except McpCallTimeoutError:
            self.breaker.failed()
            raise
        except OperationTimeoutError as exc:
            self.breaker.failed()
            raise McpCallTimeoutError() from exc
        except McpPolicyRejected as exc:
            logger.warning("mcp_policy_rejected", reasons=[reason.value for reason in exc.reasons])
            raise
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

    async def _raw_call(
        self, name: str, args: BaseModel, *, omit_unset: bool = False
    ) -> CallToolResult:
        try:
            session = await self._get_session()
            request_id = correlation_id.get()
            arguments = args.model_dump(mode="json", exclude_unset=omit_unset)
            if request_id is None:
                result = await session.call_tool(name, arguments=arguments)
            else:
                # The SDK permits extra _meta keys; mypy does not honor TypedDict extra_items.
                meta = cast("RequestParamsMeta", {"insightpilot/request_id": request_id})
                result = await session.call_tool(
                    name,
                    arguments=arguments,
                    meta=meta,
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
