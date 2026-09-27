"""Explicit source allowances and conservative whole-request accounting."""

from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import ContextBudgetExceeded
from app.core.settings_base import ConfigModel, require_configuration


class ContextSlot(StrEnum):
    """Only named context sources receive a slot allowance."""

    SYSTEM_PROMPT = "system_prompt"
    SUMMARY = "summary"
    RECENT_MESSAGES = "recent_messages"
    MEMORIES = "memories"
    SCHEMA = "schema"
    METRICS = "metrics"
    EVIDENCE = "evidence"
    DATA_RESULT = "data_result"


DEFAULT_LIMITS: dict[ContextSlot, int] = dict(
    zip(ContextSlot, (800, 500, 1500, 600, 3072, 400, 6000, 2000), strict=True)
)
TokenLimit = Annotated[int, Field(ge=1, le=2_000_000)]


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ContextPart(BaseModel):
    """A producer's rendered source, never exported in diagnostic reports."""

    model_config = ConfigDict(hide_input_in_errors=True)
    slot: ContextSlot
    text: str


class PromptBudget(BaseModel):
    """Explicit source accounting supplied alongside the actual messages.

    If system_prompt is absent, the adapter accounts for all system messages.
    Producers embedding catalog text in a system message must provide its parts.
    Other uncapped input still enters the final serialized-request bound.
    """

    parts: list[ContextPart] = Field(default_factory=list)

    def add(self, slot: ContextSlot, text: str) -> None:
        """Append a source without replacing earlier charges for that slot."""
        self.parts.append(ContextPart(slot=slot, text=text))


class ModelContextLimits(ConfigModel):
    """Operator-declared limits for one exact model, including template margins."""

    context_window: TokenLimit
    max_input_tokens: TokenLimit
    max_output_tokens: TokenLimit = 65536
    wrapper_tokens: int = Field(default=1024, ge=0, le=65536)
    per_message_tokens: int = Field(default=32, ge=0, le=1024)
    per_tool_tokens: int = Field(default=128, ge=0, le=4096)

    @model_validator(mode="after")
    def within_window(self) -> Self:
        """A declared input/output capacity cannot exceed the full window."""
        require_configuration(
            max(self.max_input_tokens, self.max_output_tokens) <= self.context_window,
            "model input/output limits exceed context window",
        )
        return self


def default_model_contexts() -> dict[str, ModelContextLimits]:
    """Frozen operator default for the approved non-thinking provider snapshot."""
    return {
        "qwen3.6-flash-2026-04-16": ModelContextLimits(
            context_window=1_000_000, max_input_tokens=991_808
        )
    }


class SlotUsage(BaseModel):
    """Safe numeric report with an explicit tokenizer label."""

    slot: ContextSlot
    used: int = Field(ge=0)
    limit: int = Field(ge=1)


class BudgetReport(BaseModel):
    """Local estimate, deliberately separate from provider-reported usage."""

    tokenizer: Literal["cl100k_base"] = "cl100k_base"
    method: Literal["utf8_upper_bound"] = "utf8_upper_bound"
    slots: list[SlotUsage]
    input_upper_bound: int = Field(ge=0)
    wrapper_reserve: int = Field(ge=0)
    output_reserve: int = Field(ge=0)
    total_upper_bound: int = Field(ge=0)
    context_window: int = Field(ge=1)
    max_input_tokens: int = Field(ge=1)


class ContextBudget:
    """Charge every occurrence; optional reduction belongs to the producer."""

    def __init__(self, counter: TokenCounter, limits: dict[ContextSlot, int] | None = None) -> None:
        self.counter = counter
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.used = dict.fromkeys(ContextSlot, 0)

    def charge(self, slot: ContextSlot, text: str) -> None:
        """Reject excess before accepting this source's additional usage."""
        count = self.used[slot] + (self.counter.count(text) if text else 0)
        if count > self.limits[slot]:
            raise ContextBudgetExceeded(slot=slot.value, used=count, limit=self.limits[slot])
        self.used[slot] = count

    def report(self) -> list[SlotUsage]:
        """Include unused slots, allowing callers to distinguish zero from missing."""
        return [SlotUsage(slot=s, used=self.used[s], limit=self.limits[s]) for s in ContextSlot]

    def check_request(
        self,
        serialized: str,
        *,
        model: ModelContextLimits,
        output_tokens: int,
        message_count: int,
        tool_count: int,
    ) -> BudgetReport:
        """Bound all input bytes plus explicit model-template and output reservations."""
        wrapper = (
            model.wrapper_tokens
            + message_count * model.per_message_tokens
            + tool_count * model.per_tool_tokens
        )
        inputs = len(serialized.encode("utf-8")) + wrapper
        total = inputs + output_tokens
        for slot, used, limit in (
            ("request_input", inputs, model.max_input_tokens),
            ("request_output", output_tokens, model.max_output_tokens),
            ("request_total", total, model.context_window),
        ):
            if used > limit:
                raise ContextBudgetExceeded(slot=slot, used=used, limit=limit)
        return BudgetReport(
            slots=self.report(),
            input_upper_bound=inputs,
            wrapper_reserve=wrapper,
            output_reserve=output_tokens,
            total_upper_bound=total,
            context_window=model.context_window,
            max_input_tokens=model.max_input_tokens,
        )
