"""Capability-driven structured output; schema errors never escape this ladder."""

import json
from pathlib import Path

import structlog
from pydantic import BaseModel, ValidationError

from app.core.errors import LlmCapabilityError, LlmResponseError, LlmStructuredOutputError
from app.services.llm.contracts import (
    Completion,
    CompletionRequest,
    GenerationBudget,
    Message,
    RepairData,
)
from app.services.llm.registry import StructuredTier
from app.services.llm.transport import LlmTransport

_PROMPTS = Path(__file__).resolve().parents[2] / "agents" / "prompts"
_JSON_PROMPT = (_PROMPTS / "structured_json.md").read_text(encoding="utf-8")
_REPAIR_PROMPT = (_PROMPTS / "structured_repair.md").read_text(encoding="utf-8")
_TOOL_NAME = "structured_response"
logger = structlog.get_logger(__name__)


async def generate_structured[T: BaseModel](
    transport: LlmTransport,
    request: CompletionRequest,
    schema: type[T],
    *,
    budget: GenerationBudget,
    starting_tier: StructuredTier,
) -> T:
    """Try each remaining tier once, then permit exactly one JSON repair call."""
    for tier in StructuredTier:
        if tier < starting_tier:
            continue
        budget.deadline.check("llm_structured")
        shaped = _shape(request, schema, tier)
        try:
            result = await transport.complete(
                shaped, deadline=budget.deadline, timeout_s=budget.timeout_s, tier=tier
            )
        except (LlmCapabilityError, LlmResponseError):
            logger.info("llm_tier_failed", model=request.model, structured_tier=tier)
            if tier is StructuredTier.PROMPTED:
                return await _repair(
                    transport,
                    shaped,
                    schema,
                    RepairData(
                        previous_output="", validation_errors="Invalid provider response envelope."
                    ),
                    budget=budget,
                )
            continue
        text = _text(result, tier)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            if tier is StructuredTier.PROMPTED:
                errors = json.dumps(
                    [
                        {"type": error["type"], "loc": error["loc"]}
                        for error in exc.errors(
                            include_input=False, include_context=False, include_url=False
                        )
                    ]
                )
                return await _repair(
                    transport,
                    shaped,
                    schema,
                    RepairData(previous_output=text, validation_errors=errors),
                    budget=budget,
                )
            logger.info("llm_tier_failed", model=request.model, structured_tier=tier)
    raise LlmStructuredOutputError()


def _shape[T: BaseModel](
    request: CompletionRequest, schema: type[T], tier: StructuredTier
) -> CompletionRequest:
    shaped = request.model_copy(deep=True)
    definition = schema.model_json_schema()
    if tier is StructuredTier.NATIVE:
        # The approved provider also requires a JSON instruction with native schemas.
        shaped.messages = [Message(role="system", content=_JSON_PROMPT), *shaped.messages]
        shaped.response_format = {
            "type": "json_schema",
            "json_schema": {"name": _TOOL_NAME, "strict": True, "schema": definition},
        }
    elif tier is StructuredTier.TOOL:
        shaped.tools = [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "parameters": definition,
                    "strict": True,
                },
            }
        ]
        shaped.tool_choice = {"type": "function", "function": {"name": _TOOL_NAME}}
        shaped.parallel_tool_calls = False
    else:
        shaped.messages = [
            Message(role="system", content=_JSON_PROMPT),
            Message(role="system", content=json.dumps(definition, ensure_ascii=False)),
            *shaped.messages,
        ]
    return shaped


def _text(result: Completion, tier: StructuredTier) -> str:
    choice = result.choices[0]
    message = choice.message
    if message.role != "assistant":
        return ""
    if tier is not StructuredTier.TOOL:
        return (
            message.content or ""
            if choice.finish_reason == "stop" and not message.tool_calls
            else ""
        )
    calls = message.tool_calls or []
    if (
        choice.finish_reason != "tool_calls"
        or len(calls) != 1
        or calls[0].function.name != _TOOL_NAME
    ):
        return ""
    return calls[0].function.arguments


async def _repair[T: BaseModel](
    transport: LlmTransport,
    request: CompletionRequest,
    schema: type[T],
    repair: RepairData,
    *,
    budget: GenerationBudget,
) -> T:
    budget.deadline.check("llm_repair")
    repaired = request.model_copy(deep=True)
    repaired.messages.extend(
        [
            Message(role="system", content=_REPAIR_PROMPT),
            Message(
                role="user",
                content=repair.model_dump_json(),
            ),
        ]
    )
    try:
        result = await transport.complete(
            repaired,
            deadline=budget.deadline,
            timeout_s=budget.timeout_s,
            tier=StructuredTier.PROMPTED,
        )
        return schema.model_validate_json(_text(result, StructuredTier.PROMPTED))
    except (ValidationError, LlmResponseError, LlmCapabilityError):
        raise LlmStructuredOutputError() from None
