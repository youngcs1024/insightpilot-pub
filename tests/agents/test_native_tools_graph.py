"""Data-only model tool calls retain the existing SQL evidence boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import ToolMessage

from app.agents.data.graph import RECURSION_LIMIT, build
from app.agents.data.state import DataAgentInput, DataAgentOutput, DataAgentState
from app.agents.failures import FailureKind
from app.agents.knowledge.graph import topology as knowledge_topology
from app.schemas.metric_resolution import MetricIntent
from app.services.graph import serializer
from app.services.llm.contracts import FunctionCall, ToolCall
from tests.agents.support import context, metric_intent, sql_candidate

if TYPE_CHECKING:
    from pydantic import BaseModel

    from app.agents.runtime import RuntimeContext
    from tests.fakes.chat_model import FakeChatModel
    from tests.fakes.mcp_client import FakeMcpClient

TWO_CALLS = 2


def _call(name: str, arguments: str, identifier: str = "native-1") -> ToolCall:
    return ToolCall(id=identifier, function=FunctionCall(name=name, arguments=arguments))


async def _invoke(
    responses: list[BaseModel | str | Exception],
) -> tuple[DataAgentOutput, RuntimeContext]:
    ctx = context(responses=responses)
    output = DataAgentOutput.model_validate(
        await build().ainvoke(
            DataAgentInput(question="2026年8月GMV", data_intent="2026年8月GMV"),
            {"recursion_limit": RECURSION_LIMIT},
            context=ctx,
        )
    )
    return output, ctx


async def test_native_tool_call_returns_tool_message_without_new_evidence() -> None:
    intent = metric_intent().model_copy(update={"native_tool_kinds": ["arithmetic"]})
    output, ctx = await _invoke(
        [
            intent,
            _call(
                "calculate_percentage",
                '{"operation":"percentage_change","first":"10","second":"15"}',
            ),
            "no_tools",
            sql_candidate(),
        ]
    )
    assert output.failure is None
    assert output.evidence.rows == [[42]]
    assert not hasattr(output.evidence, "native_tool_result")
    calls = cast("FakeChatModel", ctx.llm).calls
    assert calls[1].tool_names == ["calculate_percentage"]
    assert calls[2].tool_names == ["calculate_percentage"]
    sql_messages = calls[3].messages
    tool_messages = [item for item in sql_messages if isinstance(item, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "success"
    assert '"value": "50.0"' in tool_messages[0].content


async def test_native_tool_error_becomes_tool_message() -> None:
    intent = metric_intent().model_copy(update={"native_tool_kinds": ["arithmetic"]})
    output, ctx = await _invoke(
        [
            intent,
            _call(
                "calculate_percentage",
                '{"operation":"percentage_change","first":"0","second":"15"}',
            ),
            sql_candidate(),
        ]
    )
    assert output.failure is None
    messages = cast("FakeChatModel", ctx.llm).calls[2].messages
    errors = [item for item in messages if isinstance(item, ToolMessage)]
    assert len(errors) == 1
    assert errors[0].status == "error"
    assert "NATIVE_ARITHMETIC_UNDEFINED" in errors[0].content


async def test_invalid_native_arguments_become_error_tool_message() -> None:
    intent = metric_intent().model_copy(update={"native_tool_kinds": ["arithmetic"]})
    output, ctx = await _invoke(
        [
            intent,
            _call("calculate_percentage", '{"operation":"growth_rate","first":"10"}'),
            sql_candidate(),
        ]
    )
    assert output.failure is None
    messages = cast("FakeChatModel", ctx.llm).calls[2].messages
    errors = [item for item in messages if isinstance(item, ToolMessage)]
    assert len(errors) == 1
    assert errors[0].status == "error"
    assert "INVALID_ARGUMENTS" in errors[0].content


async def test_native_tool_cannot_bypass_sql_policy() -> None:
    intent = metric_intent().model_copy(update={"native_tool_kinds": ["arithmetic"]})
    output, ctx = await _invoke(
        [
            intent,
            _call(
                "calculate_percentage",
                '{"operation":"growth_rate","first":"10","second":"15"}',
            ),
            "no_tools",
            sql_candidate("SELECT * FROM secrets"),
        ]
    )
    assert output.failure is not None
    assert output.failure.kind is FailureKind.MCP_POLICY_REJECTED
    assert cast("FakeMcpClient", ctx.mcp).calls == []


async def test_native_tool_calls_are_bounded_and_reuse_deadline() -> None:
    intent = metric_intent().model_copy(update={"native_tool_kinds": ["periods"]})
    output, ctx = await _invoke(
        [
            intent,
            _call("resolve_period", '{"expression":"上个月"}', "native-1"),
            _call("resolve_period", '{"expression":"本月"}', "native-2"),
            sql_candidate(),
        ]
    )
    assert output.failure is None
    calls = cast("FakeChatModel", ctx.llm).calls
    assert len([call for call in calls if call.tool_names]) == TWO_CALLS
    assert len([call for call in calls if call.schema_name == "SqlGeneratorOutput"]) == 1
    assert {call.deadline_at for call in calls} == {ctx.deadline.at}


async def test_default_intent_adds_no_tool_model_call() -> None:
    output, ctx = await _invoke([MetricIntent.model_validate(metric_intent()), sql_candidate()])
    assert output.failure is None
    calls = cast("FakeChatModel", ctx.llm).calls
    assert len(calls) == TWO_CALLS
    assert all(not call.tool_names for call in calls)


def test_knowledge_graph_has_no_native_tool_node() -> None:
    assert "native_tools" not in knowledge_topology().nodes


def test_data_state_tool_exchange_round_trips_without_pickle() -> None:
    state = DataAgentState(
        question="统计GMV",
        messages=[
            ToolMessage(content='{"code":"OK"}', tool_call_id="native-1"),
        ],
    )
    codec = serializer()
    restored = codec.loads_typed(codec.dumps_typed(state))
    assert restored == state
    assert not codec.pickle_fallback
