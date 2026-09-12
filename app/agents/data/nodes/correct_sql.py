"""Bounded technical correction; neither execution nor graph assembly lives here."""

import json

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.correction_guard import preserves_semantics, same_statement
from app.agents.data.nodes.generate_sql import clean_sql, physical_tables
from app.agents.data.state import DataAgentState
from app.agents.failures import FailureKind, NodeFailure
from app.agents.prompts import SQL_CORRECT
from app.agents.runtime import RuntimeContext
from app.core.errors import DeadlineExceededError, InsightPilotError, LlmStructuredOutputError
from app.core.llm_config import ModelRole
from app.schemas.sql_correction import (
    MAX_CORRECTIONS,
    CorrectionDecision,
    CorrectionRoute,
    CorrectionStatus,
    CorrectionStopReason,
    SqlCorrectionOutput,
)

logger = structlog.get_logger(__name__)
# Policy rejection, timeout, EMPTY_RESULT and every sanity flag deliberately do
# not qualify. Transport retryability does not grant permission to rewrite SQL.
CORRECTABLE = frozenset({FailureKind.SQL_VALIDATION_FAILED, FailureKind.SQL_EXECUTION_FAILED})


def correction_route(state: DataAgentState) -> CorrectionRoute:
    """Route fresh failures only; historical failures cannot retry a successful query."""
    if state.correction_status is CorrectionStatus.TERMINAL:
        return CorrectionRoute.PACKAGE_FAILURE
    if state.query_result is not None or state.sanity_check_result.flags:
        return CorrectionRoute.NO_CORRECTION
    if state.correction_status is CorrectionStatus.PENDING_VALIDATION:
        return CorrectionRoute.VALIDATE_SQL
    if not state.failures:
        return CorrectionRoute.NO_CORRECTION
    if state.failures[-1].kind not in CORRECTABLE or state.correction_count >= MAX_CORRECTIONS:
        return CorrectionRoute.PACKAGE_FAILURE
    return CorrectionRoute.CORRECT_SQL


def should_correct(state: DataAgentState) -> bool:
    """Use typed failure and lifecycle fields, never failure detail or retryable."""
    return correction_route(state) is CorrectionRoute.CORRECT_SQL


def correction_failure(state: DataAgentState) -> NodeFailure | None:
    """Select the packager's failure, translating a spent technical-error budget.

    Step 2.11 uses this when routing directly to package_failure: exhausting the
    budget must not require a third invocation of the model or correction node.
    """
    if correction_route(state) is not CorrectionRoute.PACKAGE_FAILURE or not state.failures:
        return None
    last = state.failures[-1]
    if last.kind in CORRECTABLE and state.correction_count >= MAX_CORRECTIONS:
        return NodeFailure(
            node="correct_sql",
            kind=FailureKind.SQL_CORRECTION_EXHAUSTED,
            detail=CorrectionStopReason.BUDGET_EXHAUSTED.value,
            retryable=False,
        )
    return last


def build_messages(state: DataAgentState) -> list[BaseMessage]:
    """Keep the question → failed SQL → typed error tail; omit unrelated history."""
    return [
        SystemMessage(content=SQL_CORRECT),
        HumanMessage(
            content=json.dumps(
                {
                    "schema": state.schema_block,
                    "bindings": [
                        binding.model_dump(mode="json") for binding in state.metric_bindings
                    ],
                },
                ensure_ascii=False,
            )
        ),
        HumanMessage(content=json.dumps({"question": state.question}, ensure_ascii=False)),
        AIMessage(content=state.generated_sql, name="sql_generator"),
        HumanMessage(content=state.failures[-1].model_dump_json()),
    ]


def _stop(
    state: DataAgentState,
    reason: CorrectionStopReason,
    *,
    kind: FailureKind = FailureKind.SQL_CORRECTION_EXHAUSTED,
) -> Command[str]:
    failure = NodeFailure(node="correct_sql", kind=kind, detail=reason.value, retryable=False)
    logger.info("sql_correction_stopped", reason=reason.value, count=state.correction_count)
    return Command(
        update={
            "failures": [*state.failures, failure],
            "correction_status": CorrectionStatus.TERMINAL,
            "correction_stop_reason": reason,
        }
    )


async def correct_sql(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Return state updates for the assembler; successful candidates need validation."""
    if not should_correct(state):
        if (
            state.correction_status is CorrectionStatus.IDLE
            and state.failures
            and state.failures[-1].kind in CORRECTABLE
            and state.correction_count >= MAX_CORRECTIONS
            and correction_route(state) is CorrectionRoute.PACKAGE_FAILURE
        ):
            return _stop(state, CorrectionStopReason.BUDGET_EXHAUSTED)
        return Command(update={})
    ctx = runtime.context
    try:
        ctx.deadline.check("correct_sql")
        output = await ctx.llm.generate_structured(
            ModelRole.SQL, build_messages(state), SqlCorrectionOutput, deadline=ctx.deadline
        )
        ctx.deadline.check("correct_sql_complete")
        return _apply_candidate(state, output)
    except InsightPilotError as exc:
        logger.exception("sql_correction_failed", code=exc.code, exc_info=False)
        kinds: dict[type[InsightPilotError], FailureKind] = {
            DeadlineExceededError: FailureKind.DEADLINE_EXCEEDED,
            LlmStructuredOutputError: FailureKind.LLM_STRUCTURED_OUTPUT_FAILED,
        }
        kind = kinds.get(type(exc), FailureKind.NODE_OPERATION_FAILED)
        return _stop(state, CorrectionStopReason.OPERATION_FAILED, kind=kind)


def _apply_candidate(state: DataAgentState, output: SqlCorrectionOutput) -> Command[str]:
    if output.decision is CorrectionDecision.CANNOT_CORRECT:
        return _stop(state, CorrectionStopReason.MODEL_DECLINED)
    sql = clean_sql(output.sql)
    if sql == clean_sql(state.generated_sql) or same_statement(state.generated_sql, sql):
        return _stop(state, CorrectionStopReason.IDENTICAL_SQL)
    if not preserves_semantics(state.generated_sql, sql, state.metric_bindings):
        return _stop(state, CorrectionStopReason.SEMANTICS_UNPROVEN)
    logger.info("sql_corrected", count=state.correction_count + 1)
    return Command(
        update={
            "generated_sql": sql,
            "tables_used": physical_tables(sql),
            "messages": [*state.messages, AIMessage(content=sql, name="sql_corrector")],
            "correction_count": state.correction_count + 1,
            "correction_status": CorrectionStatus.PENDING_VALIDATION,
        }
    )
