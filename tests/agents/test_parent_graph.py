"""Actual four-route dispatch, service isolation and committed evidence barriers."""

# ruff: noqa: PLR2004 -- exact call and dispatch counts are acceptance contracts.

import asyncio
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.contracts import DataEvidence, EvidenceBundle, Route, TurnIdentity
from app.agents.graph import topology
from app.agents.state import AgentState, GraphInput
from app.core.deadline import Deadline
from app.schemas.knowledge import KnowledgeEvidence
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.retrieval import RetrievalQuery, RetrievalResult
from typing import Literal
from app.core.errors import ConflictError, McpUnavailableError, RetrievalUnavailableError
from app.services.graph import serializer
from tests.agents.parent_support import parent_context
from tests.agents.support import invoke


async def test_data_only_does_not_call_retrieval() -> None:
    ctx = parent_context(Route.DATA_ONLY)
    result = await invoke(ctx)
    assert result.status == "succeeded"
    assert len(ctx.mcp.calls) == 1
    assert ctx.retrieval.calls == []
    assert result.answer.citations == []
    assert result.evidence_refs.knowledge_snapshot_id is None


async def test_knowledge_only_does_not_call_mcp() -> None:
    ctx = parent_context(Route.KNOWLEDGE_ONLY)
    result = await invoke(ctx)
    assert result.status == "succeeded"
    assert ctx.mcp.calls == []
    assert len(ctx.retrieval.calls) == 1
    assert result.answer.citations
    assert result.answer.sql == ""
    assert result.evidence_refs.data_snapshot_id is None
    assert result.evidence_refs.knowledge_snapshot_id == ctx.evidence.knowledge.id


async def test_both_calls_both() -> None:
    ctx = parent_context(Route.BOTH)
    result = await invoke(ctx)
    assert result.status == "succeeded"
    assert len(ctx.mcp.calls) == len(ctx.retrieval.calls) == 1
    assert result.evidence_refs.data_snapshot_id and result.evidence_refs.knowledge_snapshot_id
    assert "数据结论" in result.answer.markdown
    assert "知识依据" in result.answer.markdown
    assert "尚未建立因果关系" in result.answer.markdown
    assert [call.schema_name for call in ctx.llm.calls] == [
        "RouteDecision", "MetricIntent", "SqlGeneratorOutput", "AnswerDraft", "KnowledgeDraft"
    ]


async def test_clarify_skips_specialists_and_uses_formatter() -> None:
    ctx = parent_context(Route.CLARIFY)
    updates = [item async for item in topology().compile().astream(
        GraphInput(**ctx.identity.model_dump()), context=ctx, stream_mode="updates"
    )]
    names = [name for item in updates for name in item]
    assert names == ["prepare_context", "route", "finalize_context", "clarify", "format_answer"]
    assert ctx.mcp.calls == ctx.retrieval.calls == []
    assert updates[-1]["format_answer"]["status"] == "abstained"
    assert not ctx.evidence.committed


@pytest.mark.parametrize("route", list(Route))
async def test_synthesis_only_runs_for_both(route: Route) -> None:
    ctx = parent_context(route)
    graph = topology().compile()
    names = [name async for item in graph.astream(
        GraphInput(**ctx.identity.model_dump()), context=ctx, stream_mode="updates"
    ) for name in item]
    assert names.count("synthesize") == (1 if route is Route.BOTH else 0)
    assert names.count("persist_evidence") == (0 if route is Route.CLARIFY else 1)


async def test_persistence_runs_once_after_both_and_synthesis_waits_for_commit() -> None:
    ctx = parent_context(Route.BOTH)
    data_done, knowledge_started, release_knowledge = (asyncio.Event() for _ in range(3))
    commit_started, release_commit = (asyncio.Event() for _ in range(2))
    original_query = ctx.mcp.call_tool
    original_retrieve = ctx.retrieval.retrieve
    original_commit = ctx.evidence.commit_bundle
    commits = []

    async def query(name: Literal["execute_readonly_query"], args: QueryArguments,
                    *, deadline: Deadline) -> QueryResultPayload:
        result = await original_query(name, args, deadline=deadline)
        data_done.set()
        return result

    async def retrieve(query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult:
        knowledge_started.set()
        await release_knowledge.wait()
        return await original_retrieve(query, deadline=deadline)

    async def commit(identity: TurnIdentity, data: DataEvidence | None,
                     knowledge: KnowledgeEvidence | None) -> EvidenceBundle:
        assert data_done.is_set() and release_knowledge.is_set()
        assert data is not None and knowledge is not None
        commits.append(identity)
        commit_started.set()
        await release_commit.wait()
        return await original_commit(identity, data, knowledge)

    ctx.mcp.call_tool = query
    ctx.retrieval.retrieve = retrieve
    ctx.evidence.commit_bundle = commit
    async with asyncio.timeout(10), asyncio.TaskGroup() as tasks:
        task = tasks.create_task(invoke(ctx))
        await data_done.wait()
        await knowledge_started.wait()
        assert not commit_started.is_set()
        release_knowledge.set()
        await commit_started.wait()
        assert not ctx.evidence.committed
        assert all(call.schema_name not in {"AnswerDraft", "KnowledgeDraft"}
                   for call in ctx.llm.calls)
        release_commit.set()
    assert task.result().status == "succeeded"
    assert len(commits) == 1


@pytest.mark.parametrize("missing", ["data", "knowledge", "both"])
async def test_specialist_failure_preserves_sibling_or_reports_both(missing: str) -> None:
    ctx = parent_context(
        Route.BOTH,
        data_error=McpUnavailableError() if missing in {"data", "both"} else None,
        knowledge_error=RetrievalUnavailableError() if missing in {"knowledge", "both"} else None,
    )
    result = await invoke(ctx)
    assert len(ctx.mcp.calls) == len(ctx.retrieval.calls) == 1
    if missing == "both":
        assert result.status == "failed" and result.answer is None
        assert len(result.failures) == 2
        assert not ctx.evidence.committed
    else:
        assert result.status == "degraded"
        assert missing in result.answer.degraded_components
        assert "部分回答" in result.answer.markdown


async def test_no_evidence_skips_synthesis() -> None:
    ctx = parent_context(Route.BOTH, data_error=McpUnavailableError(), empty=True)
    names = [name async for item in topology().compile().astream(
        GraphInput(**ctx.identity.model_dump()), context=ctx, stream_mode="updates"
    ) for name in item]
    assert "synthesize" not in names
    assert "format_answer" in names


async def test_knowledge_no_evidence_abstains() -> None:
    ctx = parent_context(Route.KNOWLEDGE_ONLY, empty=True)
    result = await invoke(ctx)
    assert result.status == "abstained"
    assert result.answer.abstained
    assert result.answer.evidence_refs.knowledge_snapshot_id is None
    assert ctx.mcp.calls == []


async def test_commit_failure_prevents_both_generations() -> None:
    ctx = parent_context(Route.BOTH)
    ctx.evidence.commit_bundle = AsyncMock(side_effect=ConflictError())
    result = await invoke(ctx)
    assert result.status == "failed" and result.answer is None
    assert all(call.schema_name not in {"AnswerDraft", "KnowledgeDraft"} for call in ctx.llm.calls)


async def test_production_parallel_writes_and_fresh_turn_isolation() -> None:
    graph = topology().compile(checkpointer=InMemorySaver(serde=serializer()))
    for route in (Route.BOTH, Route.DATA_ONLY, Route.KNOWLEDGE_ONLY):
        ctx = parent_context(route)
        config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
        updates = [item async for item in graph.astream(
            GraphInput(**ctx.identity.model_dump()), config, context=ctx, stream_mode="updates"
        )]
        values = {name: value for item in updates for name, value in item.items()}
        state = AgentState.model_validate((await graph.aget_state(config)).values)
        assert state.failures == []
        if route is Route.BOTH:
            overlap = set(values["data_agent"]) & set(values["knowledge_agent"])
            assert overlap == {"assumptions", "failures"}
        else:
            assert (state.knowledge_evidence is None) == (route is Route.DATA_ONLY)
            assert (state.data_evidence is None) == (route is Route.KNOWLEDGE_ONLY)
        assert state.context.memories == [] and state.context.selected_overrides.items == []


def test_graph_has_no_cycles() -> None:
    edges = topology().compile().get_graph().edges
    successors = {}
    for edge in edges:
        successors.setdefault(edge.source, set()).add(edge.target)

    def visit(node: str, ancestors: set[str]) -> None:
        assert node not in ancestors
        for child in successors.get(node, ()):
            visit(child, ancestors | {node})

    visit("__start__", set())
