"""Descriptor conversion and cached LangChain tools without network access."""

# ruff: noqa: PLR2004 -- fixed descriptor and historical fixture bounds.

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, Tool
from pydantic import BaseModel, SecretStr, ValidationError
from structlog.testing import capture_logs

from app.clients.mcp_client import McpClient
from app.clients.mcp_langchain import json_schema_to_pydantic, mcp_tool_to_langchain
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline, bind_deadline, reset_deadline
from app.core.errors import (
    McpPolicyRejected,
    McpResultError,
    McpToolSchemaError,
    McpUnavailableError,
)
from app.schemas.mcp import PolicyReason, ValidationStatus

ROOT = Path(__file__).resolve().parents[2]


def descriptor(name: str, schema: dict[str, object]) -> Tool:
    return Tool.model_validate(
        {"name": name, "description": "A discovered tool.", "inputSchema": schema}
    )


def model(schema: dict[str, object]) -> type[BaseModel]:
    return json_schema_to_pydantic(schema, name="TestArgs", tool_name="test_tool")


def budget() -> Deadline:
    return Deadline(time.monotonic() + 5)


def test_descriptor_becomes_basetool() -> None:
    client = AsyncMock()
    tool = mcp_tool_to_langchain(descriptor("probe", {"type": "object", "properties": {}}), client)
    assert isinstance(tool, BaseTool)
    assert tool.name == "probe"
    assert tool.args_schema is not None
    assert tool.args_schema.model_json_schema()["type"] == "object"


def test_args_schema_validates_required_fields() -> None:
    args = model({"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]})
    with pytest.raises(ValidationError):
        args.model_validate({})
    assert args.model_validate({"sql": "SELECT 1"}).sql == "SELECT 1"


def test_enum_converted_to_literal() -> None:
    args = model(
        {
            "type": "object",
            "$defs": {"Grain": {"type": "string", "enum": ["day", "month"]}},
            "properties": {"grain": {"$ref": "#/$defs/Grain"}},
            "required": ["grain"],
        }
    )
    assert args.model_validate({"grain": "day"}).grain == "day"
    with pytest.raises(ValidationError):
        args.model_validate({"grain": "year"})


def test_nested_object_converted() -> None:
    args = model(
        {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"column": {"type": "string"}},
                    "required": ["column"],
                }
            },
            "required": ["filter"],
        }
    )
    assert args.model_validate({"filter": {"column": "region"}}).filter.column == "region"
    with pytest.raises(ValidationError):
        args.model_validate({"filter": {"column": "region", "unknown": 1}})


def test_array_of_objects_converted() -> None:
    args = model(
        {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                }
            },
        }
    )
    assert args.model_validate({"filters": [{"name": "east"}]}).filters[0].name == "east"
    with pytest.raises(ValidationError):
        args.model_validate({"filters": [{}]})


def test_nullable_required_omitted_default_and_empty_array_stay_distinct() -> None:
    schema = {
        "type": "object",
        "properties": {
            "required_nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "optional_string": {"type": "string"},
            "with_default": {"type": "integer", "default": 3},
            "tables": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["required_nullable"],
    }
    args = model(schema)
    with pytest.raises(ValidationError):
        args.model_validate({})
    assert args.model_validate({"required_nullable": None}).required_nullable is None
    with pytest.raises(ValidationError):
        args.model_validate({"required_nullable": None, "optional_string": None})
    tool = mcp_tool_to_langchain(descriptor("probe", schema), AsyncMock())
    assert tool._parse_input({"required_nullable": None}, None) == {"required_nullable": None}
    assert tool._parse_input({"required_nullable": None, "tables": None}, None) == {
        "required_nullable": None,
        "tables": None,
    }
    assert tool._parse_input({"required_nullable": None, "tables": []}, None) == {
        "required_nullable": None,
        "tables": [],
    }


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ({"name": "ab", "rows": 1, "items": ["a"]}, True),
        ({"name": "a", "rows": 1, "items": ["a"]}, False),
        ({"name": "abcde", "rows": 1, "items": ["a"]}, False),
        ({"name": "aa", "rows": 0, "items": ["a"]}, False),
        ({"name": "aa", "rows": 11, "items": ["a"]}, False),
        ({"name": "aa", "rows": 1, "items": []}, False),
        ({"name": "aa", "rows": 1, "items": ["a", "b", "c"]}, False),
        ({"name": "aa", "rows": 1, "items": ["a"], "extra": True}, False),
    ],
)
def test_emitted_constraints_preserved(value: dict[str, object], valid: bool) -> None:
    args = model(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$"},
                "rows": {"type": "integer", "exclusiveMinimum": 0, "maximum": 10},
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 2,
                },
            },
            "required": ["name", "rows", "items"],
        }
    )
    if valid:
        assert args.model_validate(value).rows == 1
    else:
        with pytest.raises(ValidationError):
            args.model_validate(value)


def test_date_time_format_and_additional_properties_default() -> None:
    args = model(
        {"type": "object", "properties": {"at": {"type": "string", "format": "date-time"}}}
    )
    assert args.model_validate({"at": "2026-09-22T00:00:00+08:00", "extra": 1}).extra == 1
    with pytest.raises(ValidationError):
        args.model_validate({"at": "2026-09-22T00:00:00"})


def test_array_item_string_bounds_are_preserved() -> None:
    args = model(
        {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 3},
                }
            },
        }
    )
    assert args.model_validate({"tables": ["biz"]}).tables == ["biz"]
    for value in ("", "long"):
        with pytest.raises(ValidationError):
            args.model_validate({"tables": [value]})


def test_null_primitive_numeric_bounds_and_explicit_extra_permission() -> None:
    args = model(
        {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "nothing": {"type": "null"},
                "ratio": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
            },
            "required": ["nothing", "ratio"],
        }
    )
    assert args.model_validate({"nothing": None, "ratio": 0.5, "extra": 1}).extra == 1
    for value in ({"nothing": "none", "ratio": 0.5}, {"nothing": None, "ratio": 1}):
        with pytest.raises(ValidationError):
            args.model_validate(value)


def test_historical_real_descriptor_preserves_nullable_revision_bounds() -> None:
    captured = json.loads((ROOT / "tests/fixtures/get_business_schema_descriptor.json").read_text())
    assert captured["source_sha"] and captured["sdk_version"] == "2.1.1"
    tool = mcp_tool_to_langchain(Tool.model_validate(captured["tool"]), AsyncMock())
    assert tool._parse_input({}, None) == {}
    assert tool._parse_input({"known_revision": None}, None) == {"known_revision": None}
    assert tool._parse_input({"known_revision": "r"}, None) == {"known_revision": "r"}
    for value in ("", "x" * 65):
        with pytest.raises(ValidationError):
            tool._parse_input({"known_revision": value}, None)


@pytest.mark.parametrize("construct", ["oneOf", "format", "additionalProperties"])
def test_unconvertible_schema_raises_at_load(construct: str) -> None:
    bad = {
        "oneOf": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        "format": {"type": "string", "format": "uuid"},
        "additionalProperties": {"type": "object", "additionalProperties": {"type": "string"}},
    }[construct]
    schema = {"type": "object", "properties": {"bad": bad}}
    with pytest.raises(McpToolSchemaError) as caught:
        mcp_tool_to_langchain(descriptor("probe", schema), AsyncMock())
    assert caught.value.tool == "probe"
    assert construct in caught.value.construct


async def test_policy_rejection_becomes_tool_exception() -> None:
    client = AsyncMock()
    client.invoke_discovered_tool.side_effect = McpPolicyRejected(
        ValidationStatus.UNSAFE, [PolicyReason.TABLE_NOT_ALLOWED]
    )
    tool = mcp_tool_to_langchain(descriptor("probe", {"type": "object", "properties": {}}), client)
    token = bind_deadline(budget())
    try:
        with pytest.raises(ToolException):
            await tool._arun()
        output = await tool.ainvoke({})
    finally:
        reset_deadline(token)
    assert json.loads(output) == {
        "code": "MCP_POLICY_REJECTED",
        "reasons": ["table_not_allowed"],
    }


async def test_transport_failure_propagates() -> None:
    client = AsyncMock()
    client.invoke_discovered_tool.side_effect = McpUnavailableError()
    tool = mcp_tool_to_langchain(descriptor("probe", {"type": "object", "properties": {}}), client)
    token = bind_deadline(budget())
    try:
        with pytest.raises(McpUnavailableError):
            await tool.ainvoke({})
    finally:
        reset_deadline(token)


def test_sync_run_raises() -> None:
    tool = mcp_tool_to_langchain(
        descriptor("probe", {"type": "object", "properties": {}}), AsyncMock()
    )
    with pytest.raises(NotImplementedError, match="async only"):
        tool.invoke({})


async def test_refresh_rebuilds_cache_and_warns_on_change() -> None:
    session = AsyncMock(spec=ClientSession)
    first = descriptor("probe", {"type": "object", "properties": {}})
    second = descriptor("probe", {"type": "object", "properties": {"value": {"type": "string"}}})
    session.list_tools.side_effect = [
        ListToolsResult(tools=[first]),
        ListToolsResult(tools=[second]),
    ]

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        original = await client.refresh_tools(deadline=budget())
        with capture_logs() as logs:
            refreshed = await client.refresh_tools(deadline=budget())
        assert client.tools_loaded
        assert client.tools is refreshed
        assert original[0] is not refreshed[0]
        assert refreshed[0].args_schema.model_json_schema()["properties"]["value"]
        assert any(entry["event"] == "mcp_tool_descriptor_changed" for entry in logs)
    finally:
        await client.aclose()


async def test_refresh_keeps_previous_complete_cache_when_new_schema_is_invalid() -> None:
    session = AsyncMock(spec=ClientSession)
    first = descriptor("probe", {"type": "object", "properties": {}})
    bad = descriptor("new_tool", {"type": "object", "properties": {"x": {"oneOf": []}}})
    session.list_tools.side_effect = [
        ListToolsResult(tools=[first]),
        ListToolsResult(tools=[first, bad]),
    ]

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        original = await client.refresh_tools(deadline=budget())
        with pytest.raises(McpToolSchemaError, match="new_tool"):
            await client.refresh_tools(deadline=budget())
        assert client.tools is original
        assert {tool.name for tool in client.tools} == {"probe"}
    finally:
        await client.aclose()


async def test_paginated_discovery_builds_every_tool() -> None:
    session = AsyncMock(spec=ClientSession)
    first = descriptor("first", {"type": "object", "properties": {}})
    second = descriptor("second", {"type": "object", "properties": {}})
    session.list_tools.side_effect = [
        ListToolsResult(tools=[first], next_cursor="page-2"),
        ListToolsResult(tools=[second]),
    ]

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        tools = await client.refresh_tools(deadline=budget())
        assert [tool.name for tool in tools] == ["first", "second"]
        assert session.list_tools.await_args_list[1].kwargs["params"].cursor == "page-2"
    finally:
        await client.aclose()


async def test_generic_call_preserves_supplied_fields_and_validates_result() -> None:
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            descriptor(
                "future_tool",
                {
                    "type": "object",
                    "properties": {
                        "maybe": {"type": "string", "default": "server"},
                        "values": {"type": "array", "items": {"type": "string"}},
                    },
                },
            )
        ]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structured_content={"value": 1}, is_error=False
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        (tool,) = await client.refresh_tools(deadline=budget())
        token = bind_deadline(budget())
        try:
            assert json.loads(await tool.ainvoke({})) == {"value": 1}
            assert json.loads(await tool.ainvoke({"maybe": "explicit"})) == {"value": 1}
            assert json.loads(await tool.ainvoke({"values": []})) == {"value": 1}
        finally:
            reset_deadline(token)
        assert session.call_tool.await_args_list[0].kwargs["arguments"] == {}
        assert session.call_tool.await_args_list[1].kwargs["arguments"] == {"maybe": "explicit"}
        assert session.call_tool.await_args_list[2].kwargs["arguments"] == {"values": []}
    finally:
        await client.aclose()


async def test_discovered_policy_failure_is_not_retried_or_counted() -> None:
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = ListToolsResult(
        tools=[descriptor("future_tool", {"type": "object", "properties": {}})]
    )
    session.call_tool.return_value = CallToolResult(
        content=[],
        is_error=True,
        structured_content={
            "code": "MCP_POLICY_REJECTED",
            "message": "Private prose",
            "status": "unsafe",
            "reasons": ["table_not_allowed"],
        },
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        (tool,) = await client.refresh_tools(deadline=budget())
        token = bind_deadline(budget())
        try:
            result = await tool.ainvoke({})
        finally:
            reset_deadline(token)
        assert json.loads(result)["reasons"] == ["table_not_allowed"]
        assert session.call_tool.await_count == 1
        assert client.breaker.failures == 0
    finally:
        await client.aclose()


async def test_known_tool_result_is_checked_before_model_receives_it() -> None:
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = ListToolsResult(
        tools=[descriptor("get_schema", {"type": "object", "properties": {}})]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], is_error=False, structured_content={"unexpected": "value"}
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(MCPSettings(auth_token=SecretStr("test-token")), session_factory=factory)
    try:
        (tool,) = await client.refresh_tools(deadline=budget())
        token = bind_deadline(budget())
        try:
            with pytest.raises(McpResultError):
                await tool.ainvoke({})
        finally:
            reset_deadline(token)
    finally:
        await client.aclose()
