"""Specialist topology and parent projection with deterministic service doubles."""

# ruff: noqa: PLR2004 -- explicit node and bounded-attempt acceptance counts.

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import ValidationError

from app.agents.contracts import PreparedContext
from app.agents.data.graph import RECURSION_LIMIT, build
from app.agents.data.nodes import lifecycle
from app.agents.data.state import DataAgentInput, DataAgentOutput
from app.agents.failures import FailureKind, NodeFailure
from app.agents.nodes.answer_data import answer_data, answer_data_node
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.deadline import Deadline
from app.core.errors import (
    InsightPilotError,
    LlmStructuredOutputError,
    McpPolicyRejected,
    McpUnavailableError,
    SqlExecutionError,
    SqlTimeoutError,
)
from app.schemas.mcp import (
    PolicyReason,
    SqlErrorKind,
    SqlValue,
    ValidationOutcome,
    ValidationStatus,
)
from app.schemas.metric_resolution import MetricIntent
from app.schemas.sql_correction import CorrectionDecision, CorrectionStopReason, SqlCorrectionOutput
from app.services.metric_binding import build_binding
from tests.agents.correction_support import correction_context
from tests.agents.support import context, metric_intent, result, sql_candidate
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.metric_resolution_support import request, schema


async def invoke_child(
    ctx: RuntimeContext, *, callbacks: list[BaseCallbackHandler] | None = None
) -> DataAgentOutput:
    return DataAgentOutput.model_validate(
        await build().ainvoke(
            DataAgentInput(question="2026年8月的GMV", data_intent="2026年8月的GMV"),
            {"recursion_limit": RECURSION_LIMIT, "callbacks": callbacks or []},
            context=ctx,
        )
    )


def parent_state(ctx: RuntimeContext) -> AgentState:
    return AgentState(
        **ctx.identity.model_dump(),
        prepared=PreparedContext(
            question="2026年8月的GMV", summary="private-summary", messages=[], prior_sql=[]
        ),
    )


async def test_happy_path_end_to_end(fake_llm: FakeChatModel, fake_mcp: FakeMcpClient) -> None:
    fake_llm.enqueue(metric_intent(), sql_candidate())
    fake_mcp.enqueue(result())
    ctx = replace(context(), llm=fake_llm, mcp=fake_mcp)
    output = await invoke_child(ctx)
    assert output.failure is None
    assert output.evidence.rows == [[42]]
    assert output.evidence.metric_bindings[0].metric_key == "gmv"
    assert output.assumptions == output.evidence.assumptions
    assert fake_mcp.calls[0].arguments.sql == "SELECT 42"


async def test_validation_failure_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = correction_context([result()])
    original = lifecycle.SQLValidator.validate
    calls = []

    def first_failure(
        self: lifecycle.SQLValidator, sql: str, *, result_cap: int = 1000
    ) -> ValidationOutcome:
        calls.append(sql)
        if len(calls) == 1:
            # Inject a deterministic validator failure on a parseable statement:
            # unparseable originals intentionally fail Step 2.10's semantic proof.
            return ValidationOutcome(
                status=ValidationStatus.INVALID, reasons=[PolicyReason.INVALID_SQL]
            )
        return original(self, sql, result_cap=result_cap)

    monkeypatch.setattr(lifecycle.SQLValidator, "validate", first_failure)
    output = await invoke_child(ctx)
    assert output.evidence is not None
    assert output.failure is None
    assert len(calls) == 2
    assert len(ctx.mcp.calls) == 1
    assert ctx.mcp.calls[0].arguments.sql == calls[-1]


async def test_execution_failure_then_success() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_TABLE), result()])
    output = await invoke_child(ctx)
    assert output.evidence is not None
    assert output.failure is None
    assert len(ctx.mcp.calls) == 2
    assert "biz.orders" in ctx.mcp.calls[-1].arguments.sql


async def test_two_failures_then_exhausted() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_COLUMN)] * 3)
    output = await invoke_child(ctx)
    assert output.evidence is None
    assert output.failure.kind is FailureKind.SQL_CORRECTION_EXHAUSTED
    assert output.correction_stop_reason is CorrectionStopReason.BUDGET_EXHAUSTED
    assert len(ctx.mcp.calls) == 3
    assert len(ctx.llm.calls) == 4


def test_input_schema_excludes_knowledge_evidence() -> None:
    assert not {"knowledge_evidence", "messages", "memories"} & set(DataAgentInput.model_fields)
    with pytest.raises(ValidationError):
        DataAgentInput(question="GMV", knowledge_evidence={})


async def test_subgraph_output_is_typed() -> None:
    output = await invoke_child(context())
    assert DataAgentOutput.model_validate_json(output.model_dump_json()) == output
    assert "failures" not in output.model_dump()
    assert "generated_sql" not in output.model_dump()


def test_all_paths_reach_a_terminal_packager() -> None:
    edges = build().get_graph().edges
    pending = [(START, False)]
    visited = set()
    while pending:
        node, packaged = pending.pop()
        if (node, packaged) in visited:
            continue
        visited.add((node, packaged))
        packaged = packaged or node in {"package_evidence", "package_failure"}
        if node == END:
            assert packaged
        else:
            pending.extend((edge.target, packaged) for edge in edges if edge.source == node)
    assert (END, True) in visited


async def test_parent_prepared_question_reaches_subgraph() -> None:
    ctx = context()
    state = parent_state(ctx)
    output = await answer_data(state, Runtime(context=ctx), {})
    assert output.goto == "persist_evidence"
    assert state.prepared.question in str(ctx.llm.calls[0].messages)
    assert "private-summary" not in str(ctx.llm.calls)


async def test_existing_evidence_bypasses_subgraph() -> None:
    ctx = context()
    snapshot = await ctx.evidence.commit(ctx.identity, (await invoke_child(ctx)).evidence)
    replay = replace(ctx, llm=Mock(), mcp=Mock())
    output = await answer_data(parent_state(ctx), Runtime(context=replay), {})
    assert output.update["data_evidence"] == snapshot.data
    replay.llm.generate_structured.assert_not_called()
    replay.mcp.call_tool.assert_not_called()


async def test_subgraph_success_maps_to_parent_data_evidence() -> None:
    ctx = context()
    output = await answer_data(parent_state(ctx), Runtime(context=ctx), {})
    assert output.goto == "persist_evidence"
    assert output.update["data_evidence"].metric_bindings


async def test_terminal_failure_appends_without_overwriting_parent_failures() -> None:
    ctx = context(mcp_results=[McpUnavailableError()])
    state = parent_state(ctx)
    prior = NodeFailure(
        node="prepare", kind=FailureKind.NODE_OPERATION_FAILED, detail="old", retryable=False
    )
    state.failures = [prior]
    output = await answer_data(state, Runtime(context=ctx), {})
    assert state.failures == [prior]
    assert len(output.update["failures"]) == 1
    assert output.update["failures"][-1].kind is FailureKind.MCP_UNAVAILABLE


async def test_recovered_child_failure_does_not_fail_parent() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_TABLE), result()])
    output = await answer_data(parent_state(ctx), Runtime(context=ctx), {})
    assert output.goto == "persist_evidence"
    assert "failures" not in output.update


async def test_wrapper_preserves_runtime_deadline_and_checkpoint_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context()
    child = AsyncMock(
        return_value=DataAgentOutput(
            failure=NodeFailure(
                node="execute_sql", kind=FailureKind.SQL_TIMEOUT, detail="timeout", retryable=False
            )
        ).model_dump()
    )
    monkeypatch.setattr("app.agents.nodes.answer_data.DATA_GRAPH.ainvoke", child)
    config = {
        "configurable": {"thread_id": str(ctx.identity.turn_id), "checkpoint_ns": "parent:test"}
    }
    await answer_data(parent_state(ctx), Runtime(context=ctx), config)
    assert child.call_args.args[1] is config
    assert child.call_args.kwargs["context"] is ctx
    assert child.call_args.kwargs["context"].deadline is ctx.deadline


@pytest.mark.parametrize("rows", [[], [[0]], [[None]]])
async def test_empty_and_advisory_results_never_correct(rows: list[list[SqlValue]]) -> None:
    ctx = context(mcp_results=[result(rows)])
    output = await invoke_child(ctx)
    assert output.evidence is not None
    assert output.failure is None
    assert len(ctx.llm.calls) == 2
    assert len(ctx.mcp.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        SqlTimeoutError(),
        SqlExecutionError(SqlErrorKind.OTHER),
        McpUnavailableError(),
        McpPolicyRejected(ValidationStatus.UNSAFE, [PolicyReason.TABLE_NOT_ALLOWED]),
    ],
)
async def test_noncorrectable_execution_failure_stops(error: InsightPilotError) -> None:
    ctx = context(mcp_results=[error])
    output = await invoke_child(ctx)
    assert output.failure is not None
    assert output.evidence is None
    assert len(ctx.llm.calls) == 2
    assert len(ctx.mcp.calls) == 1


@pytest.mark.parametrize(
    "sql", ["DELETE FROM biz.orders", "SELECT * FROM pg_catalog.pg_user", "SELECT 1; SELECT 2"]
)
async def test_policy_precheck_never_calls_mcp(sql: str) -> None:
    ctx = context(responses=[metric_intent(), sql_candidate(sql)])
    output = await invoke_child(ctx)
    assert output.failure is not None
    assert not ctx.mcp.calls
    assert len(ctx.llm.calls) == 2


async def test_clarification_has_no_technical_failure_or_sql() -> None:
    ctx = context(responses=[MetricIntent(metric_keys=["imagined_profit"])])
    output = await invoke_child(ctx)
    assert output.clarification is not None
    assert output.failure is None
    assert output.evidence is None
    assert not ctx.mcp.calls
    assert len(ctx.llm.calls) == 1


async def test_semantic_drift_stops_before_execution() -> None:
    canonical = build_binding(request(), schema()).binding.resolved_expression
    ctx = context(
        responses=[
            metric_intent(),
            sql_candidate(canonical),
            SqlCorrectionOutput(decision=CorrectionDecision.CORRECTED, sql="SELECT 1"),
        ],
        mcp_results=[SqlExecutionError(SqlErrorKind.UNDEFINED_COLUMN)],
    )
    output = await invoke_child(ctx)
    assert output.correction_stop_reason is CorrectionStopReason.SEMANTICS_UNPROVEN
    assert output.evidence is None
    assert len(ctx.mcp.calls) == 1


@pytest.mark.parametrize("expired", [False, True])
async def test_early_failure_is_packaged(expired: bool) -> None:
    ctx = context(responses=[LlmStructuredOutputError()])
    if expired:
        ctx = replace(ctx, deadline=Deadline(0))
    output = await invoke_child(ctx)
    assert output.failure is not None
    assert output.evidence is None
    assert not ctx.mcp.calls


async def test_parent_requires_prepared_before_evidence_reuse() -> None:
    ctx = context()
    output = await answer_data(AgentState(**ctx.identity.model_dump()), Runtime(context=ctx), {})
    assert output.goto == END
    assert output.update["failures"][0].kind is FailureKind.NODE_OPERATION_FAILED
    assert not ctx.llm.calls
    assert not ctx.mcp.calls


def test_unpacked_output_is_rejected() -> None:
    with pytest.raises(ValidationError, match="terminal outcome"):
        DataAgentOutput()


async def test_registered_parent_adapter_forwards_current_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context()
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}, "tags": ["caller"]}
    wrapper = AsyncMock(return_value=Command(goto="persist_evidence"))
    monkeypatch.setattr("app.agents.nodes.answer_data.get_config", lambda: config)
    monkeypatch.setattr("app.agents.nodes.answer_data.answer_data", wrapper)
    state = parent_state(ctx)
    runtime = Runtime(context=ctx)
    await answer_data_node(state, runtime=runtime)
    assert wrapper.call_args.args == (state, runtime, config)
    assert wrapper.call_args.args[2] is config
