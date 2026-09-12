"""One active synchronous operation and bounded cancellable waiting work."""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.core.background import spawn
from model_runtime.errors import ModelDeadlineError, ModelError, ModelQueueError


@dataclass
class Work:
    """Transient request work; no durable business state lives in this queue."""

    call: Callable[[], object]
    at: float
    future: asyncio.Future[object]


class InferenceExecutor:
    """Cancellation abandons a response, never releases a running CUDA slot."""

    def __init__(self, capacity: int = 8) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ip-inference")
        self._capacity = capacity
        self._waiting: deque[Work] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def submit(self, call: Callable[[], object], at: float) -> object:
        """Admission and queue mutation are atomic on the single event loop."""
        if self._closed:
            raise ModelError()
        if at <= time.monotonic():
            raise ModelDeadlineError()
        self._waiting = deque(work for work in self._waiting if not work.future.cancelled())
        if self._worker is not None and len(self._waiting) >= self._capacity:
            raise ModelQueueError()
        work = Work(call, at, asyncio.get_running_loop().create_future())
        if self._worker is None:
            self._worker = spawn(self._drain(work), name="model-inference")
        else:
            self._waiting.append(work)
        try:
            async with asyncio.timeout_at(at):
                return await asyncio.shield(work.future)
        except TimeoutError as exc:
            work.future.cancel()
            raise ModelDeadlineError() from exc
        except asyncio.CancelledError:
            work.future.cancel()
            raise

    async def _drain(self, work: Work) -> None:
        try:
            while True:
                await self._execute(work)
                if not self._waiting:
                    break
                work = self._waiting.popleft()
        finally:
            self._worker = None

    async def _execute(self, work: Work) -> None:
        if work.future.cancelled():
            return
        if work.at <= time.monotonic():
            work.future.set_exception(ModelDeadlineError())
            return
        try:
            result = await asyncio.get_running_loop().run_in_executor(self._pool, work.call)
        except Exception as exc:
            if not work.future.done():
                work.future.set_exception(exc)
        else:
            if not work.future.done():
                work.future.set_result(result)

    async def aclose(self) -> None:
        """Stop admission and drain active GPU work before destroying the executor."""
        self._closed = True
        while self._waiting:
            work = self._waiting.popleft()
            if not work.future.done():
                work.future.set_exception(ModelError())
        if self._worker is not None:
            await asyncio.shield(self._worker)
        self._pool.shutdown(wait=False)
