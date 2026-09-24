"""Retained, observed background tasks with bounded shutdown cancellation."""

import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
_tasks: set[asyncio.Task[Any]] = set()


def _completed(task: asyncio.Task[Any]) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.exception("background_task_failed", task_name=task.get_name())


def spawn[T](coro: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]:
    """Keep a strong reference and observe failures until the task completes."""
    task = asyncio.create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_completed)
    return task


async def shutdown(timeout_s: float) -> None:
    """Cancel retained tasks and wait at most the supplied cleanup budget."""
    tasks = set(_tasks)
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    _, pending = await asyncio.wait(tasks, timeout=timeout_s)
    if pending:
        logger.warning("background_shutdown_timeout", pending=len(pending))
