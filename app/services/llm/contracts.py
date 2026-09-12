"""Minimal typed OpenAI-compatible wire contracts, independent of probe code."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.core.deadline import Deadline


class WireModel(BaseModel):
    """Hide provider payloads in validation errors."""

    model_config = ConfigDict(hide_input_in_errors=True)


class FunctionCall(WireModel):
    """A named function with JSON-encoded arguments."""

    name: str
    arguments: str


class ToolCall(WireModel):
    """Tool descriptors used in history and provider responses."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(WireModel):
    """Text-only chat history, retaining assistant tool calls and tool IDs."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class CompletionRequest(WireModel):
    """One HTTP call; shaping dictionaries stay inside the provider adapter."""

    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int
    enable_thinking: Literal[False] = False
    stream: Literal[False] = False
    response_format: dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: dict[str, JsonValue] | None = None
    parallel_tool_calls: Literal[False] | None = None


class Choice(WireModel):
    """The selected response and its termination reason."""

    message: Message
    finish_reason: str


class Usage(WireModel):
    """Only provider-reported token counts; missing usage remains unknown."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class Completion(WireModel):
    """Exactly one completion is requested and accepted."""

    choices: list[Choice] = Field(min_length=1, max_length=1)
    usage: Usage | None = None


class ProviderError(WireModel):
    """Use codes and parameter names only; provider prose is intentionally omitted."""

    code: str | None = None
    param: str | None = None


class ErrorResponse(WireModel):
    """Structured rejection envelope."""

    error: ProviderError


class GenerationBudget(WireModel):
    """The original request deadline and role-specific per-call timeout."""

    deadline: Deadline
    timeout_s: float = Field(gt=0, le=120)


class RepairData(WireModel):
    """Untrusted output and safe error categories; never exported to observability."""

    previous_output: str
    validation_errors: str
