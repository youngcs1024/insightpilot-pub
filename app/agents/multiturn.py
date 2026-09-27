"""Bounded, complete reference data for parent rewriting and SQL generation."""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agents.budget import token_bound
from app.services.context_history import trim_context_messages
from app.services.schema_tokens import SchemaTokenCounter
from app.agents.contracts import HistoryMessage

PRIOR_SQL_TOKENS = 2000


def history_cost(messages: list[BaseMessage]) -> int:
    """Bound JSON-encoded message content plus role overhead conservatively."""
    return SchemaTokenCounter().count(
        json.dumps(
            [
                {
                    "role": "user" if isinstance(message, HumanMessage) else "assistant",
                    "content": message.content,
                }
                for message in messages
            ],
            ensure_ascii=False,
        )
    )


def trim_history(messages: list[HistoryMessage]) -> list[HistoryMessage]:
    """Keep a recent complete window beginning with a user, never partial text."""
    converted: list[BaseMessage] = [
        HumanMessage(content=item.content)
        if item.role == "user"
        else AIMessage(content=item.content)
        for item in messages
    ]
    trimmed = trim_context_messages(converted, SchemaTokenCounter())
    return [
        HistoryMessage(
            role="user" if isinstance(item, HumanMessage) else "assistant",
            content=str(item.content),
        )
        for item in trimmed
    ]


def prior_queries(statements: list[str]) -> list[str]:
    """Keep at most three whole queries within the serialized reference budget."""
    selected: list[str] = []
    for statement in statements[:3]:
        candidate = [*selected, statement]
        if token_bound(json.dumps(candidate, ensure_ascii=False)) <= PRIOR_SQL_TOKENS:
            selected.append(statement)
    return selected
