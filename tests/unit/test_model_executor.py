"""Real thread/loop cancellation tests, independent of CUDA and transport timing."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import asyncio
import threading
import time

import pytest

from model_runtime.errors import ModelDeadlineError, ModelQueueError
from model_runtime.executor import InferenceExecutor


async def test_cancelled_active_work_keeps_slot_and_overflow_is_bounded() -> None:
    executor = InferenceExecutor(capacity=1)
    entered, release = threading.Event(), threading.Event()
    executed: list[str] = []

    def blocking() -> str:
        entered.set()
        assert release.wait(3)
        return "first"

    first = asyncio.create_task(executor.submit(blocking, time.monotonic() + 3))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(
            executor.submit(lambda: executed.append("second"), time.monotonic() + 3)
        )
        await asyncio.sleep(0)
        with pytest.raises(ModelQueueError):
            await executor.submit(lambda: None, time.monotonic() + 3)
        assert executed == []
        release.set()
        await second
        assert executed == ["second"]
    finally:
        release.set()
        if second:
            await asyncio.gather(second, return_exceptions=True)
        await executor.aclose()


async def test_expired_and_cancelled_queued_work_are_never_executed() -> None:
    executor = InferenceExecutor()
    entered, release = threading.Event(), threading.Event()
    executed: list[str] = []

    def blocking() -> None:
        entered.set()
        assert release.wait(3)

    active = asyncio.create_task(executor.submit(blocking, time.monotonic() + 3))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(ModelDeadlineError):
            await executor.submit(lambda: executed.append("expired"), time.monotonic() + 0.01)
        cancelled = asyncio.create_task(
            executor.submit(lambda: executed.append("cancelled"), time.monotonic() + 3)
        )
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        await active
        assert executed == []
    finally:
        release.set()
        await executor.aclose()


async def test_worker_error_does_not_poison_later_requests() -> None:
    executor = InferenceExecutor()

    def failure() -> None:
        raise ModelDeadlineError()

    try:
        with pytest.raises(ModelDeadlineError):
            await executor.submit(failure, time.monotonic() + 1)
        assert await executor.submit(lambda: 42, time.monotonic() + 1) == 42
    finally:
        await executor.aclose()
