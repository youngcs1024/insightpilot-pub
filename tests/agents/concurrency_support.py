"""Event-gated service probes around the real parent and specialist graphs."""

import asyncio
import time
from typing import Literal

import pytest

from app.agents.data.state import DataAgentOutput
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.knowledge.state import KnowledgeAgentOutput
from app.agents.runtime import RuntimeContext
from app.agents.summarize import package_result
from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.retrieval import RetrievalQuery, RetrievalResult
from tests.agents.support import result


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


def sleeping_children(
    ctx: RuntimeContext, probe: ParallelProbe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The roadmap's one-second child stubs isolate parent scheduling overhead."""
    data = DataAgentOutput(evidence=package_result(result(), []))
    knowledge = KnowledgeAgentOutput(evidence=package_evidence(ctx.retrieval.responses[0], ctx))
    # Stubbed data nodes do not consume MetricIntent or SqlGeneratorOutput.
    responses = ctx.llm._responses
    decision = responses.popleft()
    responses.popleft()
    responses.popleft()
    responses.appendleft(decision)

    async def data_child(*args: object, **kwargs: object) -> dict[str, object]:
        await probe.wait("data", ctx.deadline)
        probe.finish("data")
        return data.model_dump()

    async def knowledge_child(*args: object, **kwargs: object) -> dict[str, object]:
        await probe.wait("knowledge", ctx.deadline)
        probe.finish("knowledge")
        return knowledge.model_dump()

    monkeypatch.setattr("app.agents.nodes.answer_data.DATA_GRAPH.ainvoke", data_child)
    monkeypatch.setattr(
        "app.agents.nodes.answer_knowledge.KNOWLEDGE_GRAPH.ainvoke", knowledge_child
    )
