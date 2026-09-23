"""Absolute monotonic request budgets shared by middleware and services."""

import asyncio
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass

from fastapi import Request

from app.core.errors import DeadlineExceededError


@dataclass(frozen=True)
class Deadline:
    """One immutable deadline; child operations may shrink but never renew it."""

    at: float

    def remaining(self) -> float:
        """Return the nonnegative remaining seconds."""
        return max(0.0, self.at - time.monotonic())

    def check(self, op: str) -> None:
        """Reject work that cannot start within the request budget."""
        if self.remaining() <= 0:
            raise DeadlineExceededError("deadline exceeded before operation", operation=op)

    def budget(self, want: float) -> float:
        """Limit a call's configured timeout to the remaining request budget."""
        return min(want, self.remaining())


_current_deadline: ContextVar[Deadline | None] = ContextVar("ip_deadline", default=None)


def bind_deadline(deadline: Deadline) -> Token[Deadline | None]:
    """Bind the request's existing absolute budget for async tool adapters."""
    return _current_deadline.set(deadline)


def reset_deadline(token: Token[Deadline | None]) -> None:
    """Prevent one request's budget from leaking into the next."""
    _current_deadline.reset(token)


def current_deadline() -> Deadline:
    """Require an active request budget before a discovered tool performs I/O."""
    value = _current_deadline.get()
    if value is None:
        raise DeadlineExceededError("tool invoked without request deadline", operation="mcp_tool")
    return value


def get_deadline(request: Request) -> Deadline:
    """Inject the deadline established at the HTTP edge."""
    deadline: Deadline = request.state.deadline
    return deadline


@dataclass(frozen=True)
class ResponseBudget:
    """Only admitted chat responses may extend transport time for bounded finalization."""

    timer: asyncio.Timeout
    finalization: Deadline

    def allow_finalization(self) -> None:
        """Keep the original analysis deadline while allowing its fixed cleanup window."""
        self.timer.reschedule(self.finalization.at)
