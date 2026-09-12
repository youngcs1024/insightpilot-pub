"""Scripted implementation of the graph's read-only MCP port."""

from collections import deque
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.schema_catalog import BusinessSchemaArguments, BusinessSchemaResponse


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
        schema_responses: Sequence[BusinessSchemaResponse | Exception] = (),
    ) -> None:
        self._responses = deque(
            value.model_copy(deep=True) if isinstance(value, QueryResultPayload) else value
            for value in responses
        )
        self._calls: list[FakeMcpCall] = []
        self._schema_responses: deque[BusinessSchemaResponse | Exception] = deque()
        self._schema_calls: list[BusinessSchemaArguments] = []
        self.enqueue_schema(*schema_responses)

    def enqueue_schema(self, *responses: BusinessSchemaResponse | Exception) -> None:
        """Script metadata independently so schema reads cannot consume SQL results."""
        self._schema_responses.extend(
            value.model_copy(deep=True) if isinstance(value, BusinessSchemaResponse) else value
            for value in responses
        )

    @property
    def schema_calls(self) -> list[BusinessSchemaArguments]:
        """Return detached schema requests, including failed requests."""
        return [call.model_copy(deep=True) for call in self._schema_calls]

    async def get_business_schema(
        self, args: BusinessSchemaArguments, *, deadline: Deadline
    ) -> BusinessSchemaResponse:
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
