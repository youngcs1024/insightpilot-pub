"""Optional model-selected native assistance, bounded before SQL generation."""

import json
from typing import Literal

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import ValidationError as PydanticValidationError

from app.agents.data.state import DataAgentState
from app.agents.prompts import NATIVE_TOOLS
from app.agents.runtime import RuntimeContext
from app.agents.tools.native import definitions, invoke
from app.core.errors import InsightPilotError, LlmCapabilityError
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, observe

logger = structlog.get_logger(__name__)
MAX_NATIVE_CALLS = 2


def _request(state: DataAgentState) -> list[BaseMessage]:
    """Only resolved, typed context accompanies the untrusted question."""
    return [
        SystemMessage(content=NATIVE_TOOLS),
        HumanMessage(
            content=json.dumps(
                {
                    "question": state.data_intent or state.question,
                    "bindings": [
                        {
                            "metric_key": binding.metric_key,
                            "period_start": binding.period_start.isoformat(),
                            "period_end": binding.period_end.isoformat(),
                        }
                        for binding in state.metric_bindings
                    ],
                },
                ensure_ascii=False,
            )
        ),
    ]


def _execute(
    name: str, arguments: str, state: DataAgentState, ctx: RuntimeContext
) -> tuple[Literal["success", "error"], str]:
    """Return a safe result or fixed error code for exactly one model call."""
    try:
        with observe("native_tool", TraceMetadata(tool=name)):
            value = invoke(
                name,
                arguments,
                kinds=state.native_tool_kinds,
                now=ctx.now,
                reference_period=state.reference_period,
            )
        return "success", json.dumps({"code": "OK", "value": value}, ensure_ascii=False)
    except PydanticValidationError:
        return "error", json.dumps({"code": "INVALID_ARGUMENTS"})
    except InsightPilotError as exc:
        logger.exception("native_tool_rejected", tool=name, code=exc.code, exc_info=False)
        return "error", json.dumps({"code": exc.code})
    except Exception:
        logger.exception("native_tool_failed", tool=name)
        return "error", json.dumps({"code": "internal_error"})


async def native_tools(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Let the model call at most two pure tools without changing evidence or bindings."""
    ctx = runtime.context
    tools = definitions(state.native_tool_kinds)
    if not tools:
        return Command(update={})
    messages = _request(state)
    exchange: list[AIMessage | ToolMessage] = []
    for _ in range(MAX_NATIVE_CALLS):
        ctx.deadline.check("native_tools")
        try:
            calls = await ctx.llm.call_with_tools(
                ModelRole.SQL, messages, tools, deadline=ctx.deadline
            )
        except LlmCapabilityError:
            logger.exception("native_tool_provider_unsupported", exc_info=False)
            break
        if not calls:
            break
        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments)
        except (TypeError, ValueError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        assistant = AIMessage(
            content="",
            tool_calls=[{"id": call.id, "name": call.function.name, "args": arguments}],
        )
        ctx.deadline.check("native_tool_execute")
        status, content = _execute(call.function.name, call.function.arguments, state, ctx)
        reply = ToolMessage(content=content, tool_call_id=call.id, status=status)
        exchange.extend((assistant, reply))
        messages.extend((assistant, reply))
        if status == "error":
            break
    return Command(update={"messages": [*state.messages, *exchange]})
