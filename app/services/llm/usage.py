"""Request-local provider usage accounting, independent of tracing availability."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, Field

from app.services.llm.contracts import Usage


class UsageTotal(BaseModel):
    """Unreported HTTP attempts make the aggregate unknown, never zero."""

    attempts: int = Field(default=0, ge=0)
    reported: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int | None:
        """Count only complete provider-reported input and output usage."""
        return self.tokens if self.attempts > 0 and self.reported == self.attempts else None


_usage: ContextVar[UsageTotal | None] = ContextVar("ip_llm_usage", default=None)


@contextmanager
def collect_usage() -> Iterator[UsageTotal]:
    """Keep retries/repair together and isolate concurrent logical calls."""
    result = UsageTotal()
    token = _usage.set(result)
    try:
        yield result
    finally:
        _usage.reset(token)


def record_attempt() -> None:
    """Record an HTTP attempt before any response validation or transport failure."""
    current = _usage.get()
    if current is not None:
        current.attempts += 1


def record_usage(usage: Usage | None) -> None:
    """Attribute a validated provider response to the active collector."""
    current = _usage.get()
    if current is not None and usage is not None:
        current.reported += 1
        current.tokens += usage.prompt_tokens + usage.completion_tokens
