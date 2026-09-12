"""Translate text chat history at the provider boundary without flattening roles."""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.errors import LlmRequestError
from app.services.llm.contracts import FunctionCall, Message, ToolCall


def to_wire(messages: list[BaseMessage]) -> list[Message]:
    """Reject unsupported modalities and preserve tool relationships explicitly."""
    return [_message(message) for message in messages]


def _message(message: BaseMessage) -> Message:
    if not isinstance(message.content, str):
        raise LlmRequestError()
    if isinstance(message, SystemMessage):
        return Message(role="system", content=message.content)
    if isinstance(message, HumanMessage):
        return Message(role="user", content=message.content)
    if isinstance(message, ToolMessage):
        return Message(role="tool", content=message.content, tool_call_id=message.tool_call_id)
    if not isinstance(message, AIMessage) or message.invalid_tool_calls:
        raise LlmRequestError()
    return Message(
        role="assistant",
        content=message.content,
        tool_calls=[
            ToolCall(
                id=call["id"] or "",
                function=FunctionCall(name=call["name"], arguments=json.dumps(call["args"])),
            )
            for call in message.tool_calls
        ]
        or None,
    )
