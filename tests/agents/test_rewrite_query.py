"""Required history interpretation and no-first-turn-roundtrip acceptance."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.failures import FailureKind
from app.agents.knowledge.graph import build
from app.agents.knowledge.query_context import bounded_history
from app.agents.knowledge.state import KnowledgeAgentInput, KnowledgeAgentOutput, KnowledgeAgentState
from app.core.deadline import Deadline
from app.core.errors import LlmStructuredOutputError
from app.core.llm_config import ModelRole
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.knowledge import KnowledgeEvidence
from app.schemas.knowledge_query import KnowledgeClarificationKind, KnowledgeRewrite
from app.schemas.retrieval import PointTimeScope, RetrievalQuery
from app.services.graph import serializer
from tests.agents.knowledge_support import FakeRetrieval, inputs, invoke, ranked
from tests.agents.support import context
from tests.knowledge_query_support import rewrite, topic
from tests.observability_support import tracing


async def test_first_turn_skips_rewrite() -> None:
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx, inputs(question="七天无理由退货政策", time_scope=None))
    assert output.evidence.chunks
    assert ctx.llm.calls == []
    assert len(service.calls) == 1
    assert service.calls[0].expanded_terms == []
    assert service.calls[0].standalone == "七天无理由退货政策"


async def test_first_turn_historical_time_resolved() -> None:
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx, inputs(question="2025年8月退货政策", time_scope=None))
    assert output.evidence.time_scope.periods[0].start == date(2025, 8, 1)
    assert ctx.llm.calls == []


async def test_missing_time_disclosed() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval(ranked()), now=datetime(2026, 12, 31, 16, 30, tzinfo=UTC))
    output = await invoke(ctx, inputs(question="退货运费规则", time_scope=None))
    assert output.evidence.time_scope.as_of == date(2027, 1, 1)
    assert output.assumptions == list(output.evidence.assumptions)
    assert "Asia/Shanghai 的 2027-01-01" in output.assumptions[0]


async def test_standalone_question_does_not_inherit_history_or_call_llm() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval(ranked()))
    output = await invoke(ctx, inputs(question="促销规则是什么？", time_scope=None, knowledge_history=[topic(2024)]))
    assert output.evidence.time_scope.as_of == ctx.now.date()
    assert ctx.llm.calls == []


async def test_followup_becomes_standalone() -> None:
    prior = topic()
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[rewrite(prior)]), retrieval=service)
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[prior]))
    assert service.calls[0].standalone == "退货政策中的运费规则"
    assert service.calls[0].time_scope == prior.time_scope
    assert output.evidence.original_question == "那运费呢？"
    assert "沿用" in output.assumptions[0]
    assert len(ctx.llm.calls) == 1
    assert ctx.llm.calls[0].role is ModelRole.ROUTER


async def test_followup_explicit_time_overrides_history() -> None:
    prior = topic(2025, 7)
    service = FakeRetrieval(ranked())
    ctx = replace(context(responses=[rewrite(prior, "2026年8月退货政策中的运费规则")]), retrieval=service)
    output = await invoke(ctx, inputs(question="那2026年8月的运费呢？", time_scope=None, knowledge_history=[prior]))
    assert output.evidence.time_scope.periods[0].start == date(2026, 8, 1)
    assert output.assumptions == []


async def test_yearless_followup_preserves_history_year() -> None:
    prior = topic(2024, 7)
    ctx = replace(context(responses=[rewrite(prior, "2024年8月退货政策")]), retrieval=FakeRetrieval(ranked()))
    output = await invoke(ctx, inputs(question="那8月呢？", time_scope=None, knowledge_history=[prior]))
    assert output.evidence.time_scope.periods[0].start == date(2024, 8, 1)
    assert "2024" in output.assumptions[0]


async def test_multiple_periods_retained() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval(ranked()))
    output = await invoke(ctx, inputs(question="比较2026年7月和8月退货政策", time_scope=None))
    assert [period.label for period in output.evidence.time_scope.periods] == ["2026年7月", "2026年8月"]
    assert ctx.llm.calls == []


@pytest.mark.parametrize("question", ["2026年13月政策", "2026年2月30日政策", "2026年7月和13月政策", "去年同期政策"])
async def test_invalid_explicit_time_clarifies(question: str) -> None:
    service = FakeRetrieval()
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx, inputs(question=question, time_scope=None))
    assert output.clarification.kind is KnowledgeClarificationKind.PERIOD_UNRESOLVED
    assert output.failure is None
    assert not output.abstained
    assert service.calls == ctx.llm.calls == []


async def test_first_turn_unresolved_reference_clarifies_without_llm() -> None:
    service = FakeRetrieval()
    ctx = replace(context(responses=[]), retrieval=service)
    output = await invoke(ctx, inputs(question="那个运费呢？", time_scope=None))
    assert output.clarification.kind is KnowledgeClarificationKind.REFERENCE_UNRESOLVED
    assert service.calls == ctx.llm.calls == []


@pytest.mark.parametrize("bad_reference", ["missing", "unknown", "unresolved"])
async def test_invalid_antecedent_cannot_reach_retrieval(bad_reference: str) -> None:
    prior = topic()
    draft = rewrite(prior)
    if bad_reference == "missing":
        draft.referenced_turn_ids = []
    elif bad_reference == "unknown":
        draft.referenced_turn_ids = [uuid4()]
    else:
        draft.unresolved_references = ["运费对应哪个政策"]
    service = FakeRetrieval()
    ctx = replace(context(responses=[draft]), retrieval=service)
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[prior]))
    assert output.clarification.kind is KnowledgeClarificationKind.REFERENCE_UNRESOLVED
    assert not service.calls


async def test_conflicting_antecedent_periods_clarify() -> None:
    first, second = topic(2025), topic(2026)
    draft = KnowledgeRewrite(standalone="退货运费", referenced_turn_ids=[first.turn_id, second.turn_id])
    service = FakeRetrieval()
    ctx = replace(context(responses=[draft]), retrieval=service)
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[first, second]))
    assert output.clarification is not None
    assert service.calls == []


@pytest.mark.parametrize("draft_text", ["2025年8月退货运费规则", "2026年8月和9月退货运费规则", "2026年13月退货政策"])
async def test_rewrite_cannot_change_explicit_periods(draft_text: str) -> None:
    prior = topic()
    service = FakeRetrieval()
    ctx = replace(context(responses=[rewrite(prior, draft_text)]), retrieval=service)
    output = await invoke(ctx, inputs(question="那2026年8月运费呢？", time_scope=None, knowledge_history=[prior]))
    assert output.clarification.kind is KnowledgeClarificationKind.PERIOD_UNRESOLVED
    assert service.calls == []


async def test_supplied_scope_conflict_clarifies() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval())
    output = await invoke(ctx, inputs(question="2026年7月退款政策", time_scope=topic().time_scope))
    assert output.clarification.kind is KnowledgeClarificationKind.PERIOD_UNRESOLVED
    assert ctx.llm.calls == ctx.retrieval.calls == []


async def test_required_rewrite_failure_is_typed_and_does_not_search_raw_followup() -> None:
    service = FakeRetrieval()
    ctx = replace(context(responses=[LlmStructuredOutputError("private-secret")]), retrieval=service)
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[topic()]))
    assert output.failure.kind is FailureKind.LLM_STRUCTURED_OUTPUT_FAILED
    assert output.clarification is None
    assert service.calls == []
    assert "private-secret" not in output.model_dump_json()


async def test_expired_deadline_skips_time_rewrite_and_retrieval() -> None:
    service = FakeRetrieval()
    ctx = replace(context(responses=[]), retrieval=service, deadline=Deadline(0))
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[topic()]))
    assert output.failure.kind is FailureKind.DEADLINE_EXCEEDED
    assert service.calls == ctx.llm.calls == []


async def test_query_used_recorded_in_evidence() -> None:
    prior = topic()
    ctx = replace(context(responses=[rewrite(prior, "退货相关运费承担规则")]), retrieval=FakeRetrieval(ranked()))
    output = await invoke(ctx, inputs(question="那运费呢？", time_scope=None, knowledge_history=[prior]))
    assert output.evidence.query_used == "退货相关运费承担规则"
    assert output.evidence.original_question == "那运费呢？"
    detached = KnowledgeEvidence.model_validate_json(output.evidence.model_dump_json())
    assert detached == output.evidence
    with pytest.raises(ValidationError):
        detached.original_question = "changed"


async def test_old_evidence_keeps_original_unknown() -> None:
    output = await invoke(replace(context(responses=[]), retrieval=FakeRetrieval(ranked())))
    legacy = output.evidence.model_dump(exclude={"original_question"})
    legacy["schema_version"] = 1
    restored = KnowledgeEvidence.model_validate(legacy)
    assert restored.original_question is None
    assert restored.query_used == output.evidence.query_used
    query = RetrievalQuery(schema_version=1, standalone="legacy", time_scope=PointTimeScope(as_of=date(2026, 8, 1)))
    assert query.original_question is None
    assert query.expanded_terms == []


def test_history_contract_and_duplicate_ids() -> None:
    prior = topic()
    with pytest.raises(ValidationError):
        KnowledgeAgentInput(question="政策", knowledge_history=[prior, prior])
    with pytest.raises(ValidationError):
        prior.model_validate({**prior.model_dump(), "question": "问" * 301})
    with pytest.raises(ValidationError):
        prior.model_validate({**prior.model_dump(), "sql": "SELECT 1"})
    with pytest.raises(ValidationError):
        KnowledgeAgentInput(question="政策", knowledge_history=[topic() for _ in range(4)])


def test_history_budget_discards_whole_old_topics() -> None:
    class Counter:
        name = "cl100k_base"

        def count(self, text: str) -> int:
            return len(json.loads(text)) * 800

    history = [topic(2024), topic(2025), topic(2026)]
    bounded = bounded_history(history, Counter())
    assert bounded == history[-1:]
    bounded[0].answer_summary = "changed"
    assert history[-1].answer_summary != "changed"


async def test_checkpoint_preserves_new_query_and_clarification_contracts() -> None:
    ctx = replace(context(responses=[]), retrieval=FakeRetrieval())
    result = await invoke(ctx, inputs(question="2026年13月规则", time_scope=None))
    state = KnowledgeAgentState(question="2026年13月规则", knowledge_history=[topic()], clarification=result.clarification)
    codec = serializer()
    assert codec.loads_typed(codec.dumps_typed(state)) == state
    with pytest.raises(ValidationError):
        KnowledgeAgentOutput(clarification=result.clarification, abstained=True, abstention_reason="no")


async def test_new_time_span_and_history_are_masked() -> None:
    prior = topic()
    prior.answer_summary = "private-history-secret"
    ctx = replace(context(responses=[rewrite(prior)]), retrieval=FakeRetrieval(ranked()))
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await build().ainvoke(inputs(question="那运费呢？", time_scope=None, knowledge_history=[prior]), {"callbacks": [GraphTraceCallback()]}, context=ctx)
        service.client.flush()
        spans = exporter.get_finished_spans()
        assert any(span.name == "resolve_time_scope" for span in spans)
        assert "private-history-secret" not in json.dumps([dict(span.attributes) for span in spans])
    finally:
        await service.aclose()
