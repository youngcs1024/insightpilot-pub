"""Whole-message trimming with complete native tool exchanges."""

import json

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, trim_messages

from app.core.budget import TokenCounter
from app.services.llm.messages import to_wire


def history_text(messages: list[BaseMessage]) -> str:
    """Preserve message roles, tool arguments and result identifiers in accounting."""
    return json.dumps([m.model_dump(exclude_none=True) for m in to_wire(messages)], ensure_ascii=False)


def complete_tool_groups(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop orphaned, partial or ambiguous exchanges as one indivisible group."""
    result: list[BaseMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if isinstance(message, ToolMessage):
            continue
        if not isinstance(message, AIMessage) or not message.tool_calls:
            result.append(message)
            continue
        outputs: list[ToolMessage] = []
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            output = messages[index]
            if isinstance(output, ToolMessage):
                outputs.append(output)
            index += 1
        calls = [call.get("id") or "" for call in message.tool_calls]
        ids = [output.tool_call_id for output in outputs]
        if all(calls) and len(set(calls)) == len(calls) and sorted(calls) == sorted(ids):
            result.extend([message, *outputs])
    return result


def trim_context_messages(
    messages: list[BaseMessage], counter: TokenCounter, *, max_tokens: int = 1500
) -> list[BaseMessage]:
    """Retain a recent valid suffix without splitting messages or tool groups."""
    trimmed = trim_messages(
        complete_tool_groups(messages),
        max_tokens=max_tokens,
        token_counter=lambda items: counter.count(history_text(items)),
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    return complete_tool_groups(trimmed)
