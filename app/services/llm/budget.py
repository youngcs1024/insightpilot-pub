"""Account at the final provider boundary, including structured repairs and tools."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.messages import BaseMessage, SystemMessage

from app.core.budget import BudgetReport, ContextBudget, ContextSlot, PromptBudget, TokenCounter
from app.core.errors import LlmConfigurationError
from app.services.llm.contracts import CompletionRequest

_reports: ContextVar[list[BudgetReport] | None] = ContextVar("context_budget_reports", default=None)


def prompt_budget(  # noqa: PLR0913 -- the eight named source slots are an explicit contract.
    *,
    system_prompt: str | None = None,
    summary: str | None = None,
    recent_messages: str | None = None,
    memories: str | None = None,
    schema: str | None = None,
    metrics: str | None = None,
    evidence: str | None = None,
    data_result: str | None = None,
) -> PromptBudget:
    """Tag rendered sources explicitly, independent of their message role."""
    result = PromptBudget()
    for slot, value in zip(
        ContextSlot,
        (system_prompt, summary, recent_messages, memories, schema, metrics, evidence, data_result),
        strict=True,
    ):
        if value is not None:
            result.add(slot, value)
    return result


def complete_context(messages: list[BaseMessage], context: PromptBudget | None) -> PromptBudget:
    """Ordinary static system prompts need no producer-specific annotation."""
    result = context.model_copy(deep=True) if context is not None else PromptBudget()
    if not any(part.slot is ContextSlot.SYSTEM_PROMPT for part in result.parts):
        for message in messages:
            if isinstance(message, SystemMessage):
                result.add(ContextSlot.SYSTEM_PROMPT, str(message.content))
    return result


def check_request(request: CompletionRequest, counter: TokenCounter) -> BudgetReport:
    """Use the actual shaped request, not an earlier logical-call approximation."""
    if request.model_limits is None:
        raise LlmConfigurationError("Model context limits are required before HTTP dispatch.")
    budget = ContextBudget(counter, request.slot_limits)
    for part in request.context.parts:
        budget.charge(part.slot, part.text)
    report = budget.check_request(
        request.model_dump_json(
            exclude={"model", "temperature", "max_tokens", "stream", "enable_thinking"},
            exclude_none=True,
        ),
        model=request.model_limits,
        output_tokens=request.max_tokens,
        message_count=len(request.messages),
        tool_count=len(request.tools or []),
    )
    reports = _reports.get()
    if reports is not None:
        reports.append(report)
    return report


@contextmanager
def collect_budget_reports() -> Iterator[list[BudgetReport]]:
    """Request-local diagnostic capture; never persistent application state."""
    reports: list[BudgetReport] = []
    token = _reports.set(reports)
    try:
        yield reports
    finally:
        _reports.reset(token)
