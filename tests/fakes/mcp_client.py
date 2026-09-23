"""Scripted implementation of the graph's read-only MCP port."""

from collections import deque
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.metric_tools import MetricFragment, ResolveMetricArgs
from app.schemas.schema_tools import GetSchemaArgs, SchemaResponse


class FakeMcpCall(BaseModel):
    """Detached tool call arguments for assertions."""

    name: Literal["execute_readonly_query"]
    arguments: QueryArguments


class FakeMcpClient:
    """Consume typed results/errors once and retain immutable call snapshots."""

    def __init__(
        self,
        responses: Sequence[QueryResultPayload | Exception],
        *,
        schema_responses: Sequence[SchemaResponse | Exception] = (),
        metric_responses: Sequence[MetricFragment | Exception] = (),
    ) -> None:
        self._responses = deque(
            value.model_copy(deep=True) if isinstance(value, QueryResultPayload) else value
            for value in responses
        )
        self._calls: list[FakeMcpCall] = []
        self._schema_responses: deque[SchemaResponse | Exception] = deque()
        self._schema_calls: list[GetSchemaArgs] = []
        self.enqueue_schema(*schema_responses)
        self._metric_responses: deque[MetricFragment | Exception] = deque(metric_responses)
        self._metric_calls: list[ResolveMetricArgs] = []

    @property
    def metric_calls(self) -> list[ResolveMetricArgs]:
        """Return detached metric-validation inputs."""
        return [call.model_copy(deep=True) for call in self._metric_calls]

    def enqueue_metric(self, *responses: MetricFragment | Exception) -> None:
        """Script a validated query or a typed boundary failure."""
        self._metric_responses.extend(responses)

    async def resolve_metric(
        self, args: ResolveMetricArgs, *, deadline: Deadline
    ) -> MetricFragment:
        """Default to a faithful validated response for unrelated graph tests."""
        deadline.check("fake_mcp_metric")
        self._metric_calls.append(args.model_copy(deep=True))
        if self._metric_responses:
            response = self._metric_responses.popleft()
            if isinstance(response, Exception):
                raise response
            return response.model_copy(deep=True)
        return MetricFragment(
            select_fragment="",
            from_fragment="",
            where_fragment="",
            group_by_fragment="",
            normalized_sql=args.resolved_sql,
            normalized=False,
        )

    def enqueue_schema(self, *responses: SchemaResponse | Exception) -> None:
        """Script metadata independently so schema reads cannot consume SQL results."""
        self._schema_responses.extend(
            value.model_copy(deep=True) if isinstance(value, SchemaResponse) else value
            for value in responses
        )

    @property
    def schema_calls(self) -> list[GetSchemaArgs]:
        """Return detached schema requests, including failed requests."""
        return [call.model_copy(deep=True) for call in self._schema_calls]

    async def get_schema(self, args: GetSchemaArgs, *, deadline: Deadline) -> SchemaResponse:
        """Use the same deadline discipline and strict queue as SQL calls."""
        deadline.check("fake_mcp_schema")
        self._schema_calls.append(args.model_copy(deep=True))
        if not self._schema_responses:
            raise AssertionError("FakeMcpClient schema response queue exhausted")
        response = self._schema_responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response.model_copy(deep=True)

    def enqueue(self, *responses: QueryResultPayload | Exception) -> None:
        """Append detached typed responses to a function-scoped fake fixture."""
        self._responses.extend(
            value.model_copy(deep=True) if isinstance(value, QueryResultPayload) else value
            for value in responses
        )

    @property
    def calls(self) -> list[FakeMcpCall]:
        """Return detached history, including calls which raised scripted errors."""
        return [call.model_copy(deep=True) for call in self._calls]

    async def call_tool(
        self, name: Literal["execute_readonly_query"], args: QueryArguments, *, deadline: Deadline
    ) -> QueryResultPayload:
        """Validate deadline before consuming a scripted result."""
        deadline.check("fake_mcp")
        self._calls.append(FakeMcpCall(name=name, arguments=args).model_copy(deep=True))
        if not self._responses:
            raise AssertionError("FakeMcpClient response queue exhausted")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response
