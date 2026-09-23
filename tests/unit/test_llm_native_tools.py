"""Native tool declarations reach the provider and its calls stay typed."""

import json
from time import monotonic

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage

from app.agents.tools.native import definitions
from app.core.deadline import Deadline
from app.core.errors import LlmResponseError
from app.core.llm_config import ModelRole
from tests.llm_support import URL, response, service


async def test_model_tool_call_uses_scoped_schema_and_original_deadline(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(URL).mock(
        return_value=response(
            '{"operation":"percentage_change","first":"10","second":"15"}',
            tool_name="calculate_percentage",
        )
    )
    deadline = Deadline(monotonic() + 10)
    async with service() as llm:
        calls = await llm.call_with_tools(
            ModelRole.SQL,
            [HumanMessage(content="计算增长率")],
            definitions(["arithmetic"]),
            deadline=deadline,
        )
    assert [call.function.name for call in calls] == ["calculate_percentage"]
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == "auto"
    assert sent["parallel_tool_calls"] is False
    assert [tool["function"]["name"] for tool in sent["tools"]] == ["calculate_percentage"]
    assert sent["tools"][0]["function"]["parameters"]["required"] == [
        "operation",
        "first",
        "second",
    ]
    assert "response_format" not in sent


async def test_no_tool_provider_response_is_allowed(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(return_value=response("No calculation needed."))
    async with service() as llm:
        calls = await llm.call_with_tools(
            ModelRole.SQL,
            [HumanMessage(content="普通问题")],
            definitions(["periods"]),
            deadline=Deadline(monotonic() + 10),
        )
    assert calls == []


async def test_multiple_provider_calls_are_rejected(respx_mock: respx.MockRouter) -> None:
    reply = response("{}", tool_name="resolve_period")
    payload = json.loads(reply.content)
    payload["choices"][0]["message"]["tool_calls"].append(
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "resolve_period", "arguments": "{}"},
        }
    )
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=payload))
    async with service() as llm:
        with pytest.raises(LlmResponseError):
            await llm.call_with_tools(
                ModelRole.SQL,
                [HumanMessage(content="日期")],
                definitions(["periods"]),
                deadline=Deadline(monotonic() + 10),
            )
