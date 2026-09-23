"""Bounded, process-local MCP tool quota keyed by authenticated service principal."""

import time
from collections import deque
from collections.abc import Callable

TOOL_CALL_LIMIT = 60
TOOL_CALL_WINDOW_S = 60.0


class ToolCallLimiter:
    """Retain only the latest sixty attempts needed for an exact sliding window."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._attempts: dict[str, deque[float]] = {}

    def admit(self, caller: str) -> bool:
        """Count every attempt, including a policy rejection or rate-limited call."""
        now = self._clock()
        cutoff = now - TOOL_CALL_WINDOW_S
        attempts = self._attempts.setdefault(caller, deque())
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        allowed = len(attempts) < TOOL_CALL_LIMIT
        attempts.append(now)
        if len(attempts) > TOOL_CALL_LIMIT:
            attempts.popleft()
        return allowed
