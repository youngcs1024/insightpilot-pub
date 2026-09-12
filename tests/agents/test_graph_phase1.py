"""Four-node behavior with no network; durability is tested separately on PostgreSQL."""

import ast
import time
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.agents.contracts import AnswerDraft
from app.agents.failures import FailureKind
from app.core.deadline import Deadline
from app.core.errors import (
    ConflictError,
    InsightPilotError,
    LlmStructuredOutputError,
    McpPolicyRejected,
    McpUnavailableError,
    SqlExecutionError,
    SqlTimeoutError,
)
from app.core.llm_config import ModelRole
from app.schemas.mcp import SqlErrorKind, ValidationStatus
from tests.agents.correction_support import correction_context
from tests.agents.support import context, invoke, metric_intent, result, sql_candidate

CORRECTION_CALLS = 2
SQL_SETUP_CALLS = 2


async def test_happy_path_produces_evidence_and_answer() -> None:
    ctx = context()
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert output.answer.sql == "SELECT 42 LIMIT 1001"
    assert output.evidence_refs.data_snapshot_id == ctx.evidence.snapshot.id
    assert output.answer.markdown == "42 orders"
    assert len(ctx.mcp.calls) == 1


async def test_empty_result_is_not_a_failure() -> None:
    ctx = context(mcp_results=[result([])])
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert "no rows returned" in output.answer.markdown
    assert len(ctx.llm.calls) == SQL_SETUP_CALLS
    assert ctx.evidence.snapshot.data.sanity_flags[0].value == "empty_result"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (McpPolicyRejected(ValidationStatus.UNSAFE, []), FailureKind.MCP_POLICY_REJECTED),
        (McpUnavailableError(), FailureKind.MCP_UNAVAILABLE),
        (SqlTimeoutError(), FailureKind.SQL_TIMEOUT),
        (SqlExecutionError(), FailureKind.SQL_EXECUTION_FAILED),
    ],
)
async def test_mcp_unavailable_produces_typed_failure(
    error: InsightPilotError, kind: FailureKind
) -> None:
    ctx = context(mcp_results=[error])
    output = await invoke(ctx)
    assert output.status == "failed"
    assert output.answer is None
    assert output.failures[0].kind == kind
    assert len(ctx.mcp.calls) == 1
    assert not ctx.evidence.committed


async def test_policy_rejection_is_not_retried() -> None:
    ctx = context(mcp_results=[McpPolicyRejected(ValidationStatus.UNSAFE, [])])
    await invoke(ctx)
    assert len(ctx.mcp.calls) == 1
    assert len(ctx.llm.calls) == SQL_SETUP_CALLS


async def test_structured_output_failure() -> None:
    ctx = context(responses=[LlmStructuredOutputError()])
    output = await invoke(ctx)
    assert output.failures[0].kind == FailureKind.LLM_STRUCTURED_OUTPUT_FAILED
    assert not ctx.mcp.calls


async def test_deadline_fails_before_call() -> None:
    ctx = replace(context(), deadline=Deadline(time.monotonic() - 1))
    output = await invoke(ctx)
    assert output.failures[0].kind == FailureKind.DEADLINE_EXCEEDED
    assert not ctx.llm.calls


@pytest.mark.parametrize(
    "kind",
    [SqlErrorKind.UNDEFINED_COLUMN, SqlErrorKind.UNDEFINED_TABLE, SqlErrorKind.TYPE_MISMATCH],
)
async def test_one_correction_success(kind: SqlErrorKind) -> None:
    ctx = correction_context([SqlExecutionError(kind), result()])
    assert (await invoke(ctx)).status == "succeeded"
    assert len(ctx.mcp.calls) == CORRECTION_CALLS


async def test_correction_exhausted() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_COLUMN)] * 3)
    output = await invoke(ctx)
    assert output.status == "failed"
    assert len(ctx.mcp.calls) == 3  # noqa: PLR2004 -- original plus two corrections.
    assert output.failures[0].kind == FailureKind.SQL_CORRECTION_EXHAUSTED


async def test_snapshot_committed_before_generation() -> None:
    ctx = context()
    original = ctx.llm.generate_structured

    async def checked[T: BaseModel](
        role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline
    ) -> T:
        if role == ModelRole.SYNTHESIS:
            assert ctx.evidence.committed
            assert ctx.evidence.snapshot.data.generation_block in str(messages[1].content).replace(
                '\\"', '"'
            )
        return await original(role, messages, schema, deadline=deadline)

    ctx.llm.generate_structured = checked
    assert (await invoke(ctx)).status == "succeeded"


async def test_failed_commit_prevents_generation() -> None:
    ctx = context()

    async def fail(*args: object) -> None:
        raise ConflictError()

    ctx.evidence.commit = fail
    assert (await invoke(ctx)).answer is None
    assert len(ctx.llm.calls) == SQL_SETUP_CALLS


async def test_generation_failure_keeps_snapshot() -> None:
    ctx = context(
        responses=[
            metric_intent(),
            sql_candidate(),
            LlmStructuredOutputError(),
        ]
    )
    assert (await invoke(ctx)).answer is None
    assert ctx.evidence.committed


def test_nodes_perform_no_io_directly() -> None:
    forbidden = (
        "app.clients",
        "app.db",
        "app.repositories",
        "httpx",
        "psycopg",
        "sqlalchemy",
        "pathlib",
    )
    for path in Path("app/agents/nodes").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden), path
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith(forbidden) for alias in node.names), path


async def test_evidence_reuse_does_not_execute_sql_again() -> None:
    answer = AnswerDraft(markdown="42 orders", confidence=0.9)
    ctx = context(responses=[metric_intent(), sql_candidate(), answer, answer])
    first = await invoke(ctx)
    assert first.status == "succeeded"
    assert len(ctx.mcp.calls) == 1
    second = await invoke(ctx)
    assert second.status == "succeeded"
    assert second.evidence_refs == first.evidence_refs
    assert len(ctx.mcp.calls) == 1
