"""Data execution and terminal packaging through the existing service boundaries."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.nodes.correct_sql import correction_failure
from app.agents.data.state import DataAgentState
from app.agents.data.summarize import package_result
from app.agents.failures import FailureKind, NodeFailure
from app.agents.runtime import RuntimeContext
from app.core.errors import ConflictError
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import PolicyReason, QueryArguments, ValidationStatus
from app.schemas.sql_correction import MAX_CORRECTIONS, CorrectionStatus, CorrectionStopReason


def validate_sql(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Preflight only: mandatory server validation remains the security control."""
    runtime.context.deadline.check("validate_sql")
    outcome = SQLValidator().validate(state.generated_sql)
    if outcome.status is ValidationStatus.VALID:
        # Do not send rewritten SQL: only the server applies its cap and sentinel.
        return Command(update={})
    correctable = outcome.reasons == [PolicyReason.INVALID_SQL]
    failure = NodeFailure(
        node="validate_sql",
        kind=(
            FailureKind.SQL_VALIDATION_FAILED if correctable else FailureKind.MCP_POLICY_REJECTED
        ),
        detail=",".join(reason.value for reason in outcome.reasons),
        retryable=False,
    )
    return Command(
        update={
            "failures": [*state.failures, failure],
            "correction_status": (
                CorrectionStatus.IDLE if correctable else CorrectionStatus.TERMINAL
            ),
        },
        goto="route_correction",
    )


async def execute_sql(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Return successful empty results unchanged; transport retry belongs to MCP."""
    ctx = runtime.context
    ctx.deadline.check("execute_sql")
    result = await ctx.mcp.call_tool(
        "execute_readonly_query", QueryArguments(sql=state.generated_sql), deadline=ctx.deadline
    )
    return Command(update={"query_result": result})


def package_evidence(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Freeze the exact generation block before the parent commits the snapshot."""
    if state.query_result is None:
        raise ConflictError("missing successful query result")
    evidence = package_result(
        state.query_result, state.assumptions, sanity_result=state.sanity_check_result
    )
    evidence.metric_bindings = list(state.metric_bindings)
    evidence.sanity_flags = list(state.sanity_check_result.flags)
    return Command(update={"evidence": evidence, "failure": None})


def package_failure(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Clarification is a typed stop, never a SQL error or analytical evidence."""
    if state.clarification is not None:
        return Command(update={"evidence": None, "failure": None})
    failure = correction_failure(state)
    if failure is None:
        failure = (
            state.failures[-1]
            if state.failures
            else NodeFailure(
                node="package_failure",
                kind=FailureKind.NODE_OPERATION_FAILED,
                detail="missing terminal outcome",
                retryable=False,
            )
        )
    stop_reason = state.correction_stop_reason
    if (
        failure.kind is FailureKind.SQL_CORRECTION_EXHAUSTED
        and state.correction_count >= MAX_CORRECTIONS
    ):
        stop_reason = CorrectionStopReason.BUDGET_EXHAUSTED
    return Command(
        update={"evidence": None, "failure": failure, "correction_stop_reason": stop_reason}
    )
