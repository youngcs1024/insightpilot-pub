"""Single-operation retries; never wrap transactions or whole tool groups."""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from http import HTTPStatus

import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.deadline import Deadline
from app.core.errors import (
    DeadlineExceededError,
    InsightPilotError,
    LlmRateLimitError,
    OperationTimeoutError,
    RetryNestingError,
    UpstreamUnavailableError,
)

logger = structlog.get_logger(__name__)
_active: ContextVar[bool] = ContextVar("retry_operation_active", default=False)


def is_retryable(exc: BaseException) -> bool:
    """Use exception types and declared policy, never exception prose."""
    if isinstance(exc, LlmRateLimitError):
        return exc.retryable
    if isinstance(exc, InsightPilotError):
        return (
            isinstance(exc, UpstreamUnavailableError)
            and exc.retryable
            and not HTTPStatus.BAD_REQUEST <= exc.http_status < HTTPStatus.INTERNAL_SERVER_ERROR
        )
    return isinstance(exc, TimeoutError)


async def run_operation[T](
    operation: Callable[[], Awaitable[T]],
    *,
    deadline: Deadline,
    timeout_s: float,
    name: str,
    attempts: int = 3,
    retry_policy: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """Run one safely repeatable operation, with a bounded number of total attempts.

    Callers must establish that repeating the operation is safe. Database writes
    and transactions must not use this helper. SDK retry layers must be disabled.
    Cancellation propagates unchanged; the outer budget includes backoff.
    """
    if _active.get():
        raise RetryNestingError()
    deadline.check(name)
    token = _active.set(True)
    outer = asyncio.timeout(deadline.remaining())

    async def call() -> T:
        deadline.check(name)
        logger.info("operation_attempt", operation=name)
        try:
            async with asyncio.timeout(deadline.budget(timeout_s)):
                return await operation()
        except TimeoutError as exc:
            deadline.check(name)
            raise OperationTimeoutError(operation=name) from exc

    def before_sleep(state: RetryCallState) -> None:
        logger.info(
            "operation_retry_scheduled",
            operation=name,
            attempt=state.attempt_number,
            wait_s=state.next_action.sleep if state.next_action else 0,
        )

    retrying = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(retry_policy),
        reraise=True,
        before_sleep=before_sleep,
    )
    try:
        async with outer:
            return await retrying(call)
    except TimeoutError as exc:
        if outer.expired():
            raise DeadlineExceededError(operation=name) from exc
        raise
    finally:
        _active.reset(token)
