"""Knowledge graph contracts use scripted services and no external model calls."""

# ruff: noqa: PLR2004 -- explicit relevance thresholds and acceptance bounds.

import asyncio
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from langgraph.graph import END, START
from pydantic import ValidationError

from app.agents.failures import FailureKind
from app.agents.knowledge.graph import RECURSION_LIMIT, build
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.knowledge.state import KnowledgeAgentOutput, KnowledgeAgentState
from app.core.deadline import Deadline
from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.retrieval.config import EvidenceConfig, RetrievalConfig
from app.schemas.knowledge import KnowledgeAbstention
from app.schemas.model_runtime import ModelFailureKind
from app.schemas.retrieval import PolicyPeriod, RangeTimeScope
from app.services.graph import serializer
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelError,
    ModelInputError,
    ModelOOMError,
    ModelQueueError,
)
from tests.agents.knowledge_support import FakeRetrieval, inputs, invoke, ranked
from tests.agents.support import context
from tests.knowledge_support import retrieval
from tests.observability_support import tracing


async def test_happy_path_returns_evidence() -> None:
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx)
    assert output.evidence.chunks
    assert output.evidence.meets_floor
    assert output.evidence.chunks[0].scores.rerank == 0.8
    assert output.failure is None
    assert not output.abstained
    assert output.degraded_components == []
    assert service.deadlines == [ctx.deadline]
    assert service.deadlines[0] is ctx.deadline
    assert ctx.llm.calls == []
    assert ctx.mcp.calls == []


async def test_below_floor_abstains_not_fails() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval(ranked(0.12)))
    output = await invoke(ctx)
    assert output.abstained
    assert output.failure is None
    assert output.evidence is None
    assert ctx.llm.calls == []


async def test_abstention_reason_names_floor_and_best_score() -> None:
    result = ranked(0.12)
    result.retrieval_config.filtering.absolute_floor = 0.45
    output = await invoke(replace(context(), retrieval=FakeRetrieval(result)))
    assert "阈值=0.45" in output.abstention_reason
    assert "最高重排分数=0.12" in output.abstention_reason


async def test_retrieval_unavailable_is_failure_not_abstention() -> None:
    output = await invoke(
        replace(context(), retrieval=FakeRetrieval(RetrievalUnavailableError("private-secret")))
    )
    assert output.failure.kind is FailureKind.RETRIEVAL_UNAVAILABLE
    assert not output.abstained
    assert output.evidence is None
    assert "private-secret" not in output.model_dump_json()


async def test_rerank_unavailable_degrades_with_flag() -> None:
    result = retrieval()
    result.retrieval_config.use_rerank = True
    result.degradation = ModelFailureKind.UNAVAILABLE
    result.candidates[0].scores.rrf = 0.001
    output = await invoke(replace(context(), retrieval=FakeRetrieval(result)))
    assert output.degraded_components == ["rerank"]
    assert output.evidence.chunks
    assert output.evidence.degradation is ModelFailureKind.UNAVAILABLE
    assert output.evidence.top_rerank_score is None
    assert output.evidence.meets_floor is None
    assert output.evidence.chunks[0].scores.rerank is None
    assert output.evidence.chunks[0].scores.rrf == 0.001
    assert not output.abstained


@pytest.mark.parametrize("field", ["sql", "prior_sql", "data_evidence", "metric_override", "messages"])
def test_input_schema_excludes_sql_and_data_evidence(field: str) -> None:
    with pytest.raises(ValidationError):
        inputs(**{field: []})


@pytest.mark.parametrize("kind", ["metric_override", "format_preference", "region_focus"])
def test_input_only_accepts_finalized_terminology(kind: str) -> None:
    with pytest.raises(ValidationError):
        inputs(relevant_memories=[{"type": kind, "term": "大促", "means": "618"}])


@pytest.mark.parametrize(
    "memories",
    [
        [{"term": "x" * 51, "means": "y"}],
        [{"term": "x", "means": "y" * 201}],
        [{"term": "x", "means": "y"}] * 6,
        [{"term": "x", "means": "y", "patch": {}}],
    ],
)
def test_terminology_is_bounded_and_rejects_extra_fields(memories: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError):
        inputs(relevant_memories=memories)


async def test_query_preparation_preserves_explicit_scope_without_llm() -> None:
    scope = RangeTimeScope(periods=[
        PolicyPeriod(start=date(2026, 7, 1), end=date(2026, 8, 1), label="July"),
        PolicyPeriod(start=date(2026, 8, 1), end=date(2026, 9, 1), label="August"),
    ])
    value = inputs(
        question="原问题", knowledge_intent="独立知识问题", time_scope=scope,
        region_scope={"region_ids": [1]}, conversation_summary="private-summary",
        relevant_memories=[{"type": "terminology", "term": "大促", "means": "618"}],
    )
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx, value)
    assert service.calls[0].standalone == "独立知识问题"
    assert service.calls[0].time_scope == scope
    assert output.evidence.query_used == "独立知识问题"
    assert [period.label for period in output.evidence.time_scope.periods] == ["July", "August"]
    assert value.question == "原问题"
    assert ctx.llm.calls == []
    assert "private-summary" not in output.model_dump_json()


@pytest.mark.parametrize("intent", ["", "   "])
async def test_empty_intent_uses_original_question(intent: str) -> None:
    output = await invoke(
        replace(context(), retrieval=FakeRetrieval(ranked())), inputs(knowledge_intent=intent)
    )
    assert output.evidence.query_used == inputs().question


async def test_zero_candidates_reports_unmeasured_score() -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(retrieval([]))))
    assert output.abstained
    assert output.failure is None
    assert "阈值=0.3" in output.abstention_reason
    assert "最高重排分数=未测得" in output.abstention_reason


async def test_exact_floor_is_accepted() -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(ranked(0.3))))
    assert output.evidence.meets_floor
    assert not output.abstained


async def test_budget_exhaustion_abstains_without_generation() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval(ranked()))
    ctx.settings.retrieval.evidence = EvidenceConfig(max_tokens=1)
    output = await invoke(ctx)
    assert output.abstained
    assert output.evidence is None
    assert output.failure is None
    assert "预算不足" in output.abstention_reason
    assert "最高重排分数=0.8" in output.abstention_reason
    assert ctx.llm.calls == []


@pytest.mark.parametrize("error", [ModelError(), ModelOOMError(), ModelQueueError()])
async def test_encoding_failure_never_degrades(error: ModelError) -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(error)))
    assert output.failure.kind is FailureKind.MODEL_RUNTIME_UNAVAILABLE
    assert output.failure.retryable is error.retryable
    assert output.degraded_components == []
    assert not output.abstained


@pytest.mark.parametrize(
    "error", [ModelAuthError(), ModelContractError(), ModelInputError(), RetrievalConfigurationError()]
)
async def test_contract_auth_and_configuration_failures_do_not_degrade(error: Exception) -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(error)))
    assert output.failure.kind is FailureKind.NODE_OPERATION_FAILED
    assert not output.failure.retryable
    assert output.evidence is None
    assert output.degraded_components == []


async def test_missing_service_is_explicit_failure() -> None:
    output = await invoke(context())
    assert output.failure.kind is FailureKind.RETRIEVAL_UNAVAILABLE


async def test_expired_deadline_consumes_no_service_response() -> None:
    service = FakeRetrieval(ranked())
    output = await invoke(replace(context(), retrieval=service, deadline=Deadline(0)))
    assert output.failure.kind is FailureKind.DEADLINE_EXCEEDED
    assert service.calls == []


async def test_model_deadline_is_terminal() -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(ModelDeadlineError())))
    assert output.failure.kind is FailureKind.DEADLINE_EXCEEDED
    assert not output.abstained


async def test_cancellation_propagates() -> None:
    with pytest.raises(asyncio.CancelledError):
        await invoke(replace(context(), retrieval=FakeRetrieval(asyncio.CancelledError())))


async def test_packaging_failure_is_not_no_evidence() -> None:
    value = ranked()
    value.provenance = []
    output = await invoke(replace(context(), retrieval=FakeRetrieval(value)))
    assert output.failure.node == "package_evidence"
    assert not output.abstained
    assert output.evidence is None


def test_output_rejects_empty_evidence_success() -> None:
    empty = package_evidence(retrieval([]), context())
    with pytest.raises(ValidationError):
        KnowledgeAgentOutput(evidence=empty)


async def test_output_rejects_failure_with_abstention() -> None:
    output = await invoke(context())
    with pytest.raises(ValidationError):
        KnowledgeAgentOutput(
            failure=output.failure, abstained=True, abstention_reason="not a refusal"
        )


async def test_trace_reports_service_failure() -> None:
    ctx = replace(context(), retrieval=FakeRetrieval(RetrievalUnavailableError()))
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await build().ainvoke(inputs(), {"callbacks": [GraphTraceCallback()]}, context=ctx)
        service.client.flush()
        spans = exporter.get_finished_spans()
        for name in ("retrieve", "finish_knowledge"):
            span = next(span for span in spans if span.name == name)
            assert span.attributes["langfuse.observation.metadata.status"] == "failed"
    finally:
        await service.aclose()


async def test_consecutive_invocations_do_not_inherit_terminal_state() -> None:
    graph = build()
    service = FakeRetrieval(ranked(0.1), RetrievalUnavailableError(), ranked())
    ctx = replace(context(), retrieval=service)
    outputs = [
        KnowledgeAgentOutput.model_validate(await graph.ainvoke(inputs(), context=ctx))
        for _ in range(3)
    ]
    assert outputs[0].abstained
    assert outputs[1].failure is not None
    assert outputs[2].evidence.chunks
    assert outputs[2].failure is None
    assert not outputs[2].abstained
    assert outputs[2].abstention_reason is None


def test_output_rejects_missing_ambiguous_or_empty_outcomes() -> None:
    for fields in ({}, {"abstained": True}, {"abstention_reason": "reason"}):
        with pytest.raises(ValidationError):
            KnowledgeAgentOutput.model_validate(fields)


async def test_output_rejects_evidence_with_refusal_or_wrong_degradation() -> None:
    output = await invoke(replace(context(), retrieval=FakeRetrieval(ranked())))
    for update in (
        {"abstained": True, "abstention_reason": "reason"},
        {"degraded_components": ["rerank"]},
    ):
        with pytest.raises(ValidationError):
            KnowledgeAgentOutput.model_validate({**output.model_dump(), **update})


async def test_checkpoint_roundtrip_preserves_nested_knowledge_contracts() -> None:
    result = ranked()
    output = await invoke(replace(context(), retrieval=FakeRetrieval(result)))
    state = KnowledgeAgentState(
        **inputs(relevant_memories=[{"term": "大促", "means": "618"}]).model_dump(),
        query=result.query, retrieval_result=result, packaged=output.evidence,
        evidence=output.evidence, rejection=KnowledgeAbstention.NO_EVIDENCE,
    )
    serde = serializer()
    restored = serde.loads_typed(serde.dumps_typed(state))
    assert isinstance(restored, KnowledgeAgentState)
    assert restored == state
    assert restored.retrieval_result.candidates[0].chunk_uuid == result.candidates[0].chunk_uuid
    assert restored.evidence.chunks[0].original_text == output.evidence.chunks[0].original_text
    assert not serde.pickle_fallback
    with pytest.raises(ValidationError):
        restored.evidence.chunks[0].original_text = "changed"


def test_graph_has_bounded_recursion_and_all_paths_finish() -> None:
    graph = build()
    assert graph.config["recursion_limit"] == RECURSION_LIMIT == 32
    assert graph.checkpointer is None
    edges = graph.get_graph().edges
    pending = [(START, False)]
    visited = set()
    while pending:
        node, finished = pending.pop()
        if (node, finished) in visited:
            continue
        visited.add((node, finished))
        finished = finished or node == "finish_knowledge"
        if node == END:
            assert finished
        else:
            pending.extend((edge.target, finished) for edge in edges if edge.source == node)
    assert (END, True) in visited


def test_production_defaults_match_development_selection() -> None:
    selected = yaml.safe_load(Path("evals/datasets/retrieval/selected.yaml").read_text())
    assert selected["arm"] == "D-rerank"
    config = RetrievalConfig.model_validate(selected["config"])
    actual = RetrievalConfig()
    assert actual.model_dump(exclude={"record_arm_scores"}) == config.model_dump(
        exclude={"record_arm_scores"}
    )
    assert not actual.record_arm_scores


@pytest.mark.parametrize("score, expected", [(0.1, "abstained"), (0.8, "succeeded")])
async def test_terminal_trace_status_and_masking(score: float, expected: str) -> None:
    ctx = replace(context(), retrieval=FakeRetrieval(ranked(score)))
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await build().ainvoke(
                inputs(question="private-question"),
                {"callbacks": [GraphTraceCallback()]}, context=ctx,
            )
        service.client.flush()
        spans = exporter.get_finished_spans()
        terminal = next(span for span in spans if span.name == "finish_knowledge")
        assert terminal.attributes["langfuse.observation.metadata.status"] == expected
        assert any(span.name == "retrieve" for span in spans)
        assert "private-question" not in json.dumps([dict(span.attributes) for span in spans])
    finally:
        await service.aclose()
