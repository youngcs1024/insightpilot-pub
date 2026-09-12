"""Absolute monotonic request budgets shared by middleware and services."""

import time
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


def get_deadline(request: Request) -> Deadline:
    """Inject the deadline established at the HTTP edge."""
    deadline: Deadline = request.state.deadline
    return deadline
