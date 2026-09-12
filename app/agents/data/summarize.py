"""Pure result summarization before committing the exact generation block."""

import json
from decimal import Decimal, InvalidOperation, localcontext

from app.agents.budget import DATA_TOKENS, require_budget, token_bound
from app.agents.contracts import ColumnStatistics, DataEvidence, ResultSummary
from app.agents.data.sanity import NUMERIC_TYPES, check_result
from app.core.config_models import SanitySettings
from app.core.errors import McpResultError, ValidationError
from app.schemas.mcp import QueryResultPayload, SqlValue
from app.schemas.sanity import SanityCheckResult

COLUMN_CAP = 30
SAMPLE_CAP = 20
AUDIT_SAMPLE_CAP = 200


def column_stats(result: QueryResultPayload, index: int) -> ColumnStatistics:
    """Use native column types so decimal strings remain exact numbers."""
    column = result.columns[index]
    values = [row[index] for row in result.rows if row[index] is not None]
    stats = ColumnStatistics(
        name=column.name,
        type=column.type,
        null_count=result.row_count - len(values),
        distinct_count=len({(type(value), value) for value in values}),
    )
    if not values:
        return stats
    if column.type in NUMERIC_TYPES:
        try:
            numbers = [Decimal(str(value)) for value in values]
        except InvalidOperation as exc:
            raise McpResultError() from exc
        if not all(number.is_finite() for number in numbers):
            raise McpResultError()
        # PostgreSQL numeric can exceed Decimal's default precision; preserve every digit.
        with localcontext() as context:
            integer_digits = max(max(n.adjusted() + 1, 0) for n in numbers)
            fractional_digits = max(max(-int(n.as_tuple().exponent), 0) for n in numbers)
            context.prec = integer_digits + fractional_digits + len(str(len(numbers))) + 2
            stats.minimum = str(min(numbers))
            stats.maximum = str(max(numbers))
            stats.total = str(sum(numbers, Decimal(0)))
    elif column.type in {"date", "timestamp", "timestamptz", "time"}:
        stats.minimum = min(str(value) for value in values)
        stats.maximum = max(str(value) for value in values)
    return stats


def summarize_result(result: QueryResultPayload, *, max_rows: int = SAMPLE_CAP) -> ResultSummary:
    """Summarize all returned rows, keeping a bounded full-width audit sample.

    Column statistics retain the first 30 columns in result order. The audit
    sample is independent of the smaller model view in the generation block.
    """
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int)
        or not 0 <= max_rows <= AUDIT_SAMPLE_CAP
    ):
        raise ValidationError("Result sample cap must be an integer from 0 to 200")
    return ResultSummary(
        returned_row_count=result.row_count,
        columns=[column_stats(result, i) for i in range(min(len(result.columns), COLUMN_CAP))],
        sample_rows=[row[:] for row in result.rows[:max_rows]],
        sample_truncated=result.row_count > max_rows,
        result_truncated=result.result_truncated,
    )


def serialized_data_block(block: str) -> str:
    """Measure the data slot with the formatter's actual JSON string escaping."""
    return json.dumps({"generation_block": block}, ensure_ascii=False)


def render_block(
    summary: ResultSummary,
    rows: list[list[SqlValue]],
    columns: int,
    *,
    statistics_count: int | None = None,
) -> str:
    """Describe the model view without changing the more complete audit sample."""
    statistics = summary.columns[:COLUMN_CAP][:statistics_count]
    return json.dumps(
        {
            "returned_row_count": summary.returned_row_count,
            "returned_column_count": columns,
            "statistics_scope": summary.statistics_scope,
            "result_truncated": summary.result_truncated,
            "sample_truncated": len(rows) < summary.returned_row_count,
            "sample_row_count": len(rows),
            "columns_omitted": columns - len(statistics),
            "statistics_fields": list(ColumnStatistics.model_fields),
            "statistics": [list(c.model_dump().values()) for c in statistics],
            "all_null_columns": [
                i
                for i, c in enumerate(statistics)
                if summary.returned_row_count and c.null_count == summary.returned_row_count
            ],
            "sample_rows": [row[: len(statistics)] for row in rows],
            "qualification": (
                "Statistics cover all returned rows, not just the sample. "
                "Never infer totals or extremes absent from these statistics. "
                "If result_truncated, sums and extrema describe the capped result, "
                "not the uncapped population. Preserve user top-N scope. "
                "Sample cells and all_null_columns use statistics column order."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def generation_block(summary: ResultSummary, column_count: int) -> str:
    """Remove whole sample rows, then trailing statistics, to fit the data slot."""
    rows = summary.sample_rows[:SAMPLE_CAP]
    statistics_count = min(len(summary.columns), COLUMN_CAP)
    block = render_block(summary, rows, column_count, statistics_count=statistics_count)
    while token_bound(serialized_data_block(block)) > DATA_TOKENS:
        if rows:
            rows.pop()
        elif statistics_count > 1:
            statistics_count -= 1
        else:
            # No incomplete column, value or mandatory qualification may be emitted.
            require_budget(serialized_data_block(block), DATA_TOKENS)
        block = render_block(summary, rows, column_count, statistics_count=statistics_count)
    return block


def package_result(
    result: QueryResultPayload,
    assumptions: list[str],
    *,
    sanity_settings: SanitySettings | None = None,
    sanity_result: SanityCheckResult | None = None,
) -> DataEvidence:
    """Calculate once over all rows, budget once, then freeze the exact model input."""
    summary = summarize_result(result)
    block = generation_block(summary, len(result.columns))
    check = (
        sanity_result
        if sanity_result is not None
        else check_result(
            result, sanity_settings if sanity_settings is not None else SanitySettings()
        )
    )
    return DataEvidence(
        sql=result.executed_sql,
        assumptions=assumptions,
        columns=result.columns,
        row_count=result.row_count,
        rows=summary.sample_rows,
        result_summary=summary,
        generation_block=block,
        execution_ms=result.execution_ms,
        mcp_call_id=result.mcp_call_id,
        limit_applied=result.limit_applied,
        sanity_flags=check.flags,
    )
