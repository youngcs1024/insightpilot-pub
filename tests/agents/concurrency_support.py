"""Event-gated service probes around the real parent and specialist graphs."""

import asyncio
import time
from typing import Literal

from app.agents.runtime import RuntimeContext
from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.retrieval import RetrievalQuery, RetrievalResult


class ParallelProbe:
    """Measure service overlap without replacing LangGraph's scheduler."""

    def __init__(self, ctx: RuntimeContext, *, delay: float = 0, first: str = "") -> None:
        self.barrier = asyncio.Barrier(2)
        self.entered = {name: asyncio.Event() for name in ("data", "knowledge")}
        self.finished = {name: asyncio.Event() for name in ("data", "knowledge")}
        self.starts: dict[str, float] = {}
        self.ends: dict[str, float] = {}
        self.deadlines: dict[str, Deadline] = {}
        self.delay = delay
        self.first = first
        self.query = ctx.mcp.call_tool
        self.retrieve = ctx.retrieval.retrieve
        ctx.mcp.call_tool = self.data
        ctx.retrieval.retrieve = self.knowledge

    async def wait(self, name: str, deadline: Deadline) -> None:
        self.deadlines[name] = deadline
        self.starts[name] = time.monotonic()
        self.entered[name].set()
        await self.barrier.wait()
        if self.first and self.first != name:
            await self.finished[self.first].wait()
        await asyncio.sleep(self.delay)

    def finish(self, name: str) -> None:
        self.ends[name] = time.monotonic()
        self.finished[name].set()

    async def data(
        self, name: Literal["execute_readonly_query"], args: QueryArguments, *, deadline: Deadline
    ) -> QueryResultPayload:
        try:
            await self.wait("data", deadline)
            return await self.query(name, args, deadline=deadline)
        finally:
            self.finish("data")

    async def knowledge(self, query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult:
        try:
            await self.wait("knowledge", deadline)
            return await self.retrieve(query, deadline=deadline)
        finally:
            self.finish("knowledge")
