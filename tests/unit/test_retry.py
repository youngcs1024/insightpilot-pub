"""Retry only safe typed failures, within one shared deadline."""

import asyncio
from time import monotonic

import pytest
from tenacity import wait_none

from app.core.deadline import Deadline
from app.core.errors import (
    ConflictError,
    DatabaseTimeoutError,
    DeadlineExceededError,
    InsightPilotError,
    UpstreamUnavailableError,
)
from app.core.retry import OperationTimeoutError, RetryNestingError, run_operation

ATTEMPTS = 3


class PolicyRejectedError(InsightPilotError):
    """Representative MCP adapter failure; no production MCP client exists yet."""

    code = "MCP_POLICY_REJECTED"
    http_status = 403


class UpstreamBadRequestError(UpstreamUnavailableError):
    """Even a retryable upstream subtype cannot retry a 4xx."""

    http_status = 429


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())


async def test_retries_on_upstream_unavailable(no_backoff: None) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < ATTEMPTS:
            raise UpstreamUnavailableError()
        return "ok"

    assert (
        await run_operation(operation, deadline=Deadline(monotonic() + 1), timeout_s=1, name="test")
        == "ok"
    )
    assert calls == ATTEMPTS


@pytest.mark.parametrize(
    "error",
    [
        PolicyRejectedError(),
        DeadlineExceededError(),
        DatabaseTimeoutError(),
        ConflictError(),
        UpstreamBadRequestError(),
        InsightPilotError(),
    ],
)
async def test_does_not_retry_typed_rejections(error: InsightPilotError) -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        await run_operation(operation, deadline=Deadline(monotonic() + 1), timeout_s=1, name="test")
    assert calls == 1


async def test_does_not_retry_on_policy_rejected() -> None:
    await test_does_not_retry_typed_rejections(PolicyRejectedError())


async def test_does_not_retry_on_deadline_exceeded() -> None:
    await test_does_not_retry_typed_rejections(DeadlineExceededError())


async def test_attempt_limit(no_backoff: None) -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise UpstreamUnavailableError()

    with pytest.raises(UpstreamUnavailableError):
        await run_operation(operation, deadline=Deadline(monotonic() + 1), timeout_s=1, name="test")
    assert calls == ATTEMPTS


async def test_deadline_in_backoff() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise UpstreamUnavailableError()

    with pytest.raises(DeadlineExceededError):
        await run_operation(
            operation, deadline=Deadline(monotonic() + 0.02), timeout_s=1, name="test"
        )
    assert calls == 1


async def test_single_call_timeout_retries(no_backoff: None) -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    with pytest.raises(OperationTimeoutError):
        await run_operation(
            operation, deadline=Deadline(monotonic() + 1), timeout_s=0.01, name="test"
        )
    assert calls == ATTEMPTS


async def test_deadline_cancels_inflight_call() -> None:
    cancelled = asyncio.Event()

    async def operation() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(DeadlineExceededError):
        await run_operation(
            operation, deadline=Deadline(monotonic() + 0.02), timeout_s=1, name="test"
        )
    assert cancelled.is_set()


async def test_external_cancellation_propagates() -> None:
    async def operation() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_operation(operation, deadline=Deadline(monotonic() + 1), timeout_s=1, name="test")


async def test_nested_retry_rejected() -> None:
    deadline = Deadline(monotonic() + 1)

    async def inner() -> None:
        pytest.fail("nested operation must not start")

    async def outer() -> None:
        await run_operation(inner, deadline=deadline, timeout_s=1, name="inner")

    with pytest.raises(RetryNestingError):
        await run_operation(outer, deadline=deadline, timeout_s=1, name="outer")
