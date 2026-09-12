"""Metadata and SQL scripts are independent, detached and strict."""

import pytest

from app.schemas.mcp import QueryArguments
from app.schemas.schema_catalog import BusinessSchemaArguments
from tests.agents.support import context
from tests.factories import business_schema, query_result
from tests.fakes.mcp_client import FakeMcpClient


async def test_schema_reads_cannot_consume_sql_responses() -> None:
    ctx = context()
    schema = business_schema()
    fake = FakeMcpClient([query_result()], schema_responses=[schema])
    result = await fake.get_business_schema(BusinessSchemaArguments(), deadline=ctx.deadline)
    assert result == schema
    assert result is not schema
    assert not fake.calls
    await fake.call_tool(
        "execute_readonly_query", QueryArguments(sql="SELECT 1"), deadline=ctx.deadline
    )
    assert len(fake.calls) == len(fake.schema_calls) == 1
    with pytest.raises(AssertionError, match="schema response queue exhausted"):
        await fake.get_business_schema(BusinessSchemaArguments(), deadline=ctx.deadline)
    assert len(fake.calls) == 1


async def test_schema_requests_are_detached_from_caller_and_history() -> None:
    ctx = context()
    fake = FakeMcpClient([], schema_responses=[business_schema()])
    args = BusinessSchemaArguments()
    await fake.get_business_schema(args, deadline=ctx.deadline)
    assert fake.schema_calls[0] == args
    assert fake.schema_calls[0] is not args
    assert fake.schema_calls[0] is not fake.schema_calls[0]
