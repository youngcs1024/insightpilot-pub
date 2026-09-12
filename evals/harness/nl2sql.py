"""Production data graph adapter and independent semantic scoring."""

import asyncio

from app.agents.data.graph import RECURSION_LIMIT, build
from app.agents.data.state import DataAgentInput, DataAgentState
from app.agents.runtime import RuntimeContext
from app.core.errors import InsightPilotError
from app.services.metric_patch_sql import canonical_filters
from evals.harness.contracts import Case, Observation


def bindings_match(case: Case, observation: Observation) -> bool:
    """Match each complete binding, never a union of unrelated metrics' filters."""
    bindings = {binding.metric_key: binding for binding in observation.bindings}
    if len(bindings) != len(observation.bindings) or set(bindings) != {
        expected.metric_key for expected in case.expected_bindings
    }:
        return False
    for expected in case.expected_bindings:
        actual = bindings[expected.metric_key]
        if (
            actual.period_start != expected.period.start
            or actual.period_end != expected.period.end
            or actual.date_field != expected.date_field
            or set(canonical_filters(actual.filters_applied))
            != set(canonical_filters(expected.filters))
        ):
            return False
    return True


async def observe(question: str, ctx: RuntimeContext) -> Observation:
    """Run the unchanged specialist; capture full state before the evidence sample boundary."""
    state = DataAgentState(question=question)
    error: str | None = None
    try:
        async with asyncio.timeout(ctx.deadline.remaining()):
            async for value in build().astream(
                DataAgentInput(question=question),
                context=ctx,
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode="values",
                output_keys=list(DataAgentState.model_fields),
            ):
                state = DataAgentState.model_validate(value)
    except InsightPilotError as exc:
        error = exc.code
    except TimeoutError:
        error = "EVALUATION_DEADLINE"
    if state.clarification is not None:
        error = state.clarification.kind.value
    elif state.failure is not None:
        error = state.failure.kind.value
    return Observation(
        bindings=state.metric_bindings,
        result=state.query_result,
        sql=state.generated_sql,
        corrections=state.correction_count,
        failure_code=error,
    )
