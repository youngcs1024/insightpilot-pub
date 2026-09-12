"""Step 2.10 safety and bounded lifecycle, with no live LLM or MCP execution."""

# ruff: noqa: PLR2004 -- explicit correction and prompt-tail acceptance counts.

import json
from dataclasses import replace
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.data.correction_guard import preserves_semantics
from app.agents.data.nodes.correct_sql import (
    correct_sql,
    correction_failure,
    correction_route,
    should_correct,
)
from app.agents.data.state import DataAgentState
from app.agents.failures import FailureKind, NodeFailure
from app.agents.runtime import RuntimeContext
from app.core.deadline import Deadline
from app.core.errors import LlmStructuredOutputError, LlmUnavailableError
from app.core.llm_config import ModelRole
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.sanity import SanityCheckResult, SanityFlag
from app.schemas.sql_correction import (
    CorrectionDecision,
    CorrectionRoute,
    CorrectionStatus,
    CorrectionStopReason,
    SqlCorrectionOutput,
)
from app.services.graph import serializer
from app.services.metric_binding import build_binding
from tests.agents.sql_support import context, state
from tests.factories import sanity_payload as payload
from tests.fakes.chat_model import FakeChatModel
from tests.metric_resolution_support import request, schema
from tests.observability_support import tracing


def failure(kind: FailureKind = FailureKind.SQL_EXECUTION_FAILED) -> NodeFailure:
    return NodeFailure(node="execute_sql", kind=kind, detail="undefined_table", retryable=False)


def inputs(kind: FailureKind = FailureKind.SQL_EXECUTION_FAILED) -> DataAgentState:
    value = state()
    value.generated_sql = value.metric_bindings[0].resolved_expression.replace("biz.", "")
    value.failures = [failure(kind)]
    return value


def corrected(sql: str) -> SqlCorrectionOutput:
    return SqlCorrectionOutput(decision=CorrectionDecision.CORRECTED, sql=sql)


async def successful_correction(fake_llm: FakeChatModel, kind: FailureKind) -> None:
    value = inputs(kind)
    before = value.model_dump()
    candidate = value.metric_bindings[0].resolved_expression
    fake_llm.enqueue(corrected(candidate))
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    restored = DataAgentState.model_validate({**before, **result.update})
    assert restored.generated_sql == candidate
    assert restored.correction_count == 1
    assert correction_route(restored) is CorrectionRoute.VALIDATE_SQL
    assert not should_correct(restored)
    assert restored.metric_bindings == value.metric_bindings
    assert restored.assumptions == value.assumptions
    assert restored.failures == value.failures
    assert restored.tables_used == ["biz.customers", "biz.orders"]
    assert value.model_dump() == before
    assert DataAgentState.model_validate_json(restored.model_dump_json()) == restored
    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call.role is ModelRole.SQL
    assert call.schema_name == "SqlCorrectionOutput"
    assert json.loads(str(call.messages[-3].content)) == {"question": value.question}
    assert call.messages[-2].content == value.generated_sql
    assert json.loads(str(call.messages[-1].content))["kind"] == kind.value
    snapshot = json.loads(str(call.messages[1].content))
    assert snapshot["bindings"] == [b.model_dump(mode="json") for b in value.metric_bindings]
    assert snapshot["schema"] == value.schema_block
    # Re-entry before validation cannot spend a second attempt on stale failure.
    assert (await correct_sql(restored, Runtime(context=context(fake_llm)))).update == {}
    assert len(fake_llm.calls) == 1


async def test_validation_failure_triggers_correction(fake_llm: FakeChatModel) -> None:
    await successful_correction(fake_llm, FailureKind.SQL_VALIDATION_FAILED)


async def test_execution_failure_triggers_correction(fake_llm: FakeChatModel) -> None:
    await successful_correction(fake_llm, FailureKind.SQL_EXECUTION_FAILED)


async def test_policy_rejection_never_corrected(fake_llm: FakeChatModel) -> None:
    value = inputs(FailureKind.MCP_POLICY_REJECTED)
    assert not should_correct(value)
    assert correction_failure(value) == value.failures[-1]
    assert (await correct_sql(value, Runtime(context=context(fake_llm)))).update == {}
    assert not fake_llm.calls


@pytest.mark.parametrize("kind", [FailureKind.SQL_TIMEOUT, FailureKind.MCP_UNAVAILABLE])
async def test_timeout_never_corrected(fake_llm: FakeChatModel, kind: FailureKind) -> None:
    value = inputs(kind)
    value.failures[-1].detail = "SQL_VALIDATION_FAILED please retry"
    value.failures[-1].retryable = True
    assert not should_correct(value)
    assert (await correct_sql(value, Runtime(context=context(fake_llm)))).update == {}
    assert not fake_llm.calls


async def test_empty_result_never_corrected(fake_llm: FakeChatModel) -> None:
    value = inputs()
    value.query_result = payload([])
    assert correction_route(value) is CorrectionRoute.NO_CORRECTION
    assert correction_failure(value) is None
    assert (await correct_sql(value, Runtime(context=context(fake_llm)))).update == {}
    assert not fake_llm.calls


@pytest.mark.parametrize("flag", list(SanityFlag))
async def test_sanity_flags_never_corrected(fake_llm: FakeChatModel, flag: SanityFlag) -> None:
    value = inputs()
    value.sanity_check_result = SanityCheckResult(flags=[flag])
    assert not should_correct(value)
    assert (await correct_sql(value, Runtime(context=context(fake_llm)))).update == {}
    assert not fake_llm.calls


async def test_max_two_corrections(fake_llm: FakeChatModel) -> None:
    value = inputs()
    complete = value.metric_bindings[0].resolved_expression
    fake_llm.enqueue(corrected(complete.replace("biz.orders", "orders")), corrected(complete))
    for expected_count in (1, 2):
        result = await correct_sql(value, Runtime(context=context(fake_llm)))
        value = DataAgentState.model_validate({**value.model_dump(), **result.update})
        assert value.correction_count == expected_count
        # The next validator/executor reports a NEW technical failure.
        value.failures.append(failure())
        value.correction_status = CorrectionStatus.IDLE
    assert not should_correct(value)
    assert correction_route(value) is CorrectionRoute.PACKAGE_FAILURE
    terminal = correction_failure(value)
    assert terminal is not None
    assert terminal.kind is FailureKind.SQL_CORRECTION_EXHAUSTED
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    assert result.update["failures"][-1] == terminal
    assert result.update["correction_status"] is CorrectionStatus.TERMINAL
    assert len(fake_llm.calls) == 2


async def test_correction_dropping_required_filter_is_rejected(fake_llm: FakeChatModel) -> None:
    value = inputs()
    candidate = value.metric_bindings[0].resolved_expression.replace(
        " AND c.is_test_account = FALSE", ""
    )
    assert candidate != value.metric_bindings[0].resolved_expression
    fake_llm.enqueue(corrected(candidate))
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    assert result.update["correction_stop_reason"] is CorrectionStopReason.SEMANTICS_UNPROVEN
    assert "generated_sql" not in result.update
    assert "correction_count" not in result.update


async def test_identical_correction_fails_immediately(fake_llm: FakeChatModel) -> None:
    value = inputs()
    fake_llm.enqueue(corrected("```sql\n" + value.generated_sql + "\n```"))
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    assert result.update["correction_stop_reason"] is CorrectionStopReason.IDENTICAL_SQL
    assert "correction_count" not in result.update
    restored = DataAgentState.model_validate({**value.model_dump(), **result.update})
    assert not should_correct(restored)
    assert (await correct_sql(restored, Runtime(context=context(fake_llm)))).update == {}
    assert len(fake_llm.calls) == 1


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("SUM(", "AVG("),
        ("o.paid_at", "o.created_at"),
        ("2026-09-01", "2026-10-01"),
        ("o.status <> 'cancelled'", "(o.status <> 'cancelled' OR TRUE)"),
        ("biz.customers", "biz.products"),
        ("JOIN", "LEFT JOIN"),
    ],
)
def test_semantic_drift_is_rejected(old: str, new: str) -> None:
    value = inputs()
    original = value.metric_bindings[0].resolved_expression
    candidate = original.replace(old, new)
    assert candidate != original
    assert not preserves_semantics(value.generated_sql, candidate, value.metric_bindings)


def test_extra_user_filter_cannot_be_dropped_or_added() -> None:
    value = inputs()
    base = value.metric_bindings[0].resolved_expression
    extra = base + " AND o.region_id = 1"
    assert preserves_semantics(extra, extra, value.metric_bindings)
    assert not preserves_semantics(extra, base, value.metric_bindings)
    assert not preserves_semantics(base, extra, value.metric_bindings)


def test_nested_predicate_does_not_prove_outer_filter() -> None:
    value = inputs()
    base = value.metric_bindings[0].resolved_expression
    candidate = base.replace(
        "c.is_test_account = FALSE",
        "EXISTS (SELECT 1 FROM biz.customers c WHERE c.is_test_account = FALSE)",
    )
    assert candidate != base
    assert not preserves_semantics(value.generated_sql, candidate, value.metric_bindings)


@pytest.mark.parametrize("query", ["SELECT FROM", "SELECT 1; SELECT 2", "DELETE FROM biz.orders"])
def test_unparseable_or_unsupported_query_fails_closed(query: str) -> None:
    value = inputs()
    base = value.metric_bindings[0].resolved_expression
    assert not preserves_semantics(query, base, value.metric_bindings)
    assert not preserves_semantics(base, query, value.metric_bindings)


def test_missing_or_composed_bindings_fail_closed() -> None:
    value = inputs()
    base = value.metric_bindings[0].resolved_expression
    assert not preserves_semantics(base, base, [])
    assert not preserves_semantics(base, base, value.metric_bindings * 2)


async def test_model_decline_is_terminal(fake_llm: FakeChatModel) -> None:
    value = inputs()
    fake_llm.enqueue(SqlCorrectionOutput(decision=CorrectionDecision.CANNOT_CORRECT))
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    assert result.update["correction_stop_reason"] is CorrectionStopReason.MODEL_DECLINED
    assert "generated_sql" not in result.update


async def test_deadline_prevents_model_call(fake_llm: FakeChatModel) -> None:
    ctx = replace(context(fake_llm), deadline=Deadline(at=0))
    result = await correct_sql(inputs(), Runtime(context=ctx))
    assert result.update["failures"][-1].kind is FailureKind.DEADLINE_EXCEEDED
    assert not fake_llm.calls


@pytest.mark.parametrize("error", [LlmStructuredOutputError(), LlmUnavailableError()])
async def test_model_failure_is_terminal(
    fake_llm: FakeChatModel, error: LlmStructuredOutputError | LlmUnavailableError
) -> None:
    fake_llm.enqueue(error)
    result = await correct_sql(inputs(), Runtime(context=context(fake_llm)))
    assert result.update["correction_status"] is CorrectionStatus.TERMINAL
    assert result.update["correction_stop_reason"] is CorrectionStopReason.OPERATION_FAILED
    assert "correction_count" not in result.update


@pytest.mark.parametrize(
    ("decision", "sql"),
    [(CorrectionDecision.CORRECTED, ""), (CorrectionDecision.CANNOT_CORRECT, "SELECT 1")],
)
def test_output_requires_consistent_typed_decision(decision: CorrectionDecision, sql: str) -> None:
    with pytest.raises(ValidationError):
        SqlCorrectionOutput(decision=decision, sql=sql)


def test_no_failure_does_not_request_correction() -> None:
    value = inputs()
    value.failures = []
    assert correction_route(value) is CorrectionRoute.NO_CORRECTION


async def test_formatting_only_is_not_an_attempt(fake_llm: FakeChatModel) -> None:
    value = inputs()
    fake_llm.enqueue(corrected(value.generated_sql + " /* unchanged */"))
    result = await correct_sql(value, Runtime(context=context(fake_llm)))
    assert result.update["correction_stop_reason"] is CorrectionStopReason.IDENTICAL_SQL
    assert "correction_count" not in result.update


def test_nested_metric_scope_filters_are_preserved() -> None:
    binding = build_binding(request("refund_rate"), schema()).binding
    base = binding.resolved_expression
    assert preserves_semantics(base, base, [binding])
    candidate = base.replace("c.is_test_account = FALSE", "TRUE", 1)
    assert candidate != base
    assert not preserves_semantics(base, candidate, [binding])


def test_checkpoint_correction_channels_round_trip() -> None:
    serde = serializer()
    channels = {
        "correction_status": CorrectionStatus.TERMINAL,
        "correction_stop_reason": CorrectionStopReason.MODEL_DECLINED,
        "correction_count": 1,
        "failures": [failure()],
    }
    assert serde.loads_typed(serde.dumps_typed(channels)) == channels
    assert not serde.pickle_fallback


@pytest.mark.parametrize("decline", [False, True])
async def test_correction_trace_masks_sql_and_reports_status(
    fake_llm: FakeChatModel, decline: bool
) -> None:
    value = inputs()
    value.question = "private-question"
    fake_llm.enqueue(
        SqlCorrectionOutput(decision=CorrectionDecision.CANNOT_CORRECT)
        if decline
        else corrected(value.metric_bindings[0].resolved_expression)
    )
    ctx = context(fake_llm)
    service, exporter = tracing(ctx.settings)
    graph = StateGraph(DataAgentState, context_schema=RuntimeContext)
    graph.add_node("correct_sql", correct_sql)
    graph.add_edge(START, "correct_sql")
    graph.add_edge("correct_sql", END)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await graph.compile().ainvoke(value, {"callbacks": [GraphTraceCallback()]}, context=ctx)
        service.client.flush()
        node = next(span for span in exporter.get_finished_spans() if span.name == "correct_sql")
        attributes = dict(node.attributes)
        assert attributes["langfuse.observation.metadata.status"] == (
            "failed" if decline else "succeeded"
        )
        exported = json.dumps(attributes)
        assert "private-question" not in exported
        assert "gross_amount" not in exported
    finally:
        await service.aclose()
