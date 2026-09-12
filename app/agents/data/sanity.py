"""Check all returned rows without changing query meaning or requesting execution."""

from collections import Counter
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation

import structlog

from app.core.config_models import SanitySettings
from app.core.errors import McpResultError
from app.core.observability import TraceMetadata, update_current_observation
from app.schemas.mcp import QueryResultPayload, SqlValue
from app.schemas.sanity import SanityCheckResult, SanityFlag

logger = structlog.get_logger(__name__)
NUMERIC_TYPES = frozenset({"int2", "int4", "int8", "numeric", "float4", "float8", "decimal"})


def numeric_value(value: SqlValue) -> Decimal | None:
    """Decode only values from native numeric columns, excluding booleans."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise McpResultError() from exc
    if not number.is_finite():
        raise McpResultError()
    return number


def numeric_flags(result: QueryResultPayload, settings: SanitySettings) -> Iterator[SanityFlag]:
    """Unambiguous configured aliases supply business meaning; types alone do not."""
    names = Counter(column.name for column in result.columns)
    for index, column in enumerate(result.columns):
        if column.type not in NUMERIC_TYPES:
            continue
        money = names[column.name] == 1 and column.name in settings.money_columns
        nonzero = names[column.name] == 1 and column.name in settings.nonzero_columns
        for row in result.rows:
            number = numeric_value(row[index])
            if number is not None:
                yield from value_flags(number, settings.extreme_magnitude, money, nonzero)


def value_flags(
    number: Decimal, threshold: Decimal, money: bool, nonzero: bool
) -> Iterator[SanityFlag]:
    """Exact comparisons avoid both float loss and Decimal context rounding."""
    if number.copy_abs() > threshold:
        yield SanityFlag.EXTREME_MAGNITUDE
    if money and number < 0:
        yield SanityFlag.NEGATIVE_MONEY
    if nonzero and number == 0:
        yield SanityFlag.SUSPICIOUS_ZERO


def result_flags(result: QueryResultPayload, settings: SanitySettings) -> Iterator[SanityFlag]:
    """Yield observations incrementally so later check failures cannot erase them."""
    if not result.row_count:
        yield SanityFlag.EMPTY_RESULT
    elif any(all(row[i] is None for row in result.rows) for i in range(len(result.columns))):
        yield SanityFlag.ALL_NULL
    if result.rows == [[None]]:
        yield SanityFlag.SINGLE_NULL_SCALAR
    if result.result_truncated:
        yield SanityFlag.TRUNCATED
    if settings.expected_max_rows is not None and result.row_count > settings.expected_max_rows:
        yield SanityFlag.CARDINALITY_SPIKE
    yield from numeric_flags(result, settings)


def check_result(result: QueryResultPayload | None, settings: SanitySettings) -> SanityCheckResult:
    """Best-effort inspection; neither observations nor check errors cause retries."""
    outcome = SanityCheckResult()
    try:
        if result is None:
            raise McpResultError()
        for flag in result_flags(result, settings):
            if flag not in outcome.flags:
                outcome.flags.append(flag)
    except Exception:
        # CancelledError/BaseException must propagate; only the advisory check is protected.
        outcome.check_failed = True
        logger.exception("sanity_check_failed")
    record_check(outcome)
    return outcome


def record_check(outcome: SanityCheckResult) -> None:
    """Only enum observations and check status enter logs or the current node span."""
    try:
        logger.info(
            "sanity_checked",
            sanity_flags=[flag.value for flag in outcome.flags],
            sanity_check_failed=outcome.check_failed,
        )
        update_current_observation(
            TraceMetadata(
                sanity_flags=outcome.flags,
                sanity_check_failed=outcome.check_failed,
            )
        )
    except Exception:
        logger.exception("sanity_trace_failed")
