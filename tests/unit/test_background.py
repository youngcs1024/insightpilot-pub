"""Background task lifetime, exception observation and bounded shutdown."""

import asyncio

import pytest
import structlog

from app.core import background
from app.core.errors import InsightPilotError


async def test_task_reference_retained_until_done() -> None:
    release = asyncio.Event()
    task = background.spawn(release.wait(), name="retained")
    assert task in background._tasks
    release.set()
    assert await task
    await asyncio.sleep(0)
    assert task not in background._tasks


async def test_failure_observed_and_released() -> None:
    async def fail() -> None:
        raise InsightPilotError("synthetic")

    with structlog.testing.capture_logs() as logs:
        task = background.spawn(fail(), name="failed")
        with pytest.raises(InsightPilotError):
            await task
        await asyncio.sleep(0)
    assert task not in background._tasks
    assert any(event["event"] == "background_task_failed" for event in logs)


async def test_shutdown_cancels_and_releases() -> None:
    task = background.spawn(asyncio.Event().wait(), name="cancelled")
    await background.shutdown(0.1)
    assert task.cancelled()
    assert task not in background._tasks


async def test_shutdown_is_bounded_for_slow_cleanup() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    task = background.spawn(slow(), name="slow-cleanup")
    await started.wait()
    await background.shutdown(0.01)
    assert task in background._tasks
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert task not in background._tasks
