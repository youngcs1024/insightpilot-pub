"""Sequential attempts with isolated deadlines and no added transport retries."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

import structlog

from app.core.errors import InsightPilotError
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import QueryResultPayload, ValidationStatus
from evals.harness.compare import compare
from evals.harness.contracts import (
    Case,
    CaseResult,
    ConfigSnapshot,
    Observation,
    Options,
    Report,
)
from evals.harness.nl2sql import bindings_match
from evals.metrics import summarize

type Generate = Callable[[str], Awaitable[Observation]]
type Execute = Callable[[str], Awaitable[QueryResultPayload]]
logger = structlog.get_logger(__name__)


async def evaluate(case: Case, repeat: int, generate: Generate, execute: Execute) -> CaseResult:
    """An invalid oracle invalidates evidence, while model failures remain measured failures."""
    result = CaseResult(
        case_id=case.id,
        repeat=repeat,
        traps=list(case.traps),
        adversarial=bool(case.adversarial_sql),
    )
    if case.adversarial_sql:
        outcome = SQLValidator().validate(case.adversarial_sql)
        result.policy_reasons = outcome.reasons
        result.unsafe_sql_blocked = outcome.status is not ValidationStatus.VALID and set(
            case.expected_reasons
        ).issubset(outcome.reasons)
        return result
    try:
        canonical = await execute(case.canonical_sql)
    except InsightPilotError as exc:
        result.evidence_valid = False
        result.failure_code = "canonical_" + exc.code
        return result
    result.canonical_result = canonical
    if canonical.result_truncated:
        result.evidence_valid = False
        result.failure_code = "canonical_truncated"
        return result
    if case.anchor_rows is not None and not compare(
        canonical,
        canonical.model_copy(update={"rows": case.anchor_rows, "row_count": len(case.anchor_rows)}),
        case.comparison,
        case.tolerance,
    ):
        result.evidence_valid = False
        result.failure_code = "canonical_anchor_mismatch"
        return result
    try:
        observation = await generate(case.question)
        result.observation = observation
        result.failure_code = observation.failure_code
        result.metric_resolution_accuracy = bindings_match(case, observation)
        result.execution_match = observation.result is not None
        result.result_accuracy = observation.result is not None and compare(
            observation.result, canonical, case.comparison, case.tolerance
        )
    except InsightPilotError as exc:
        result.failure_code = exc.code
    return result


async def run_suite(
    cases: list[Case],
    options: Options,
    config: ConfigSnapshot,
    generate: Generate,
    execute: Execute,
) -> Report:
    """Run all declared attempts exactly once, preserving each outcome and denominator."""
    results = []
    for repeat in range(1, options.repeats + 1):
        for case in cases:
            result = await evaluate(case, repeat, generate, execute)
            results.append(result)
            logger.info(
                "evaluation_case_completed",
                case_id=case.id,
                repeat=repeat,
                result_accuracy=result.result_accuracy,
                evidence_valid=result.evidence_valid,
                failure_code=result.failure_code,
            )
    return Report(
        run_id=uuid4().hex,
        created_at=datetime.now(UTC),
        config=config,
        repeats=options.repeats,
        expected_attempts=len(cases) * options.repeats,
        results=results,
        summary=summarize(results),
        per_trap={
            trap: summarize([r for r in results if trap in r.traps])
            for trap in sorted({trap for case in cases for trap in case.traps})
        },
        per_repeat={
            repeat: summarize([r for r in results if r.repeat == repeat])
            for repeat in range(1, options.repeats + 1)
        },
        evidence_valid=bool(results) and all(r.evidence_valid for r in results),
    )
