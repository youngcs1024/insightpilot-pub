"""Compare typed execution results, never SQL text or generation samples."""

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from app.schemas.mcp import QueryResultPayload, SqlValue
from evals.harness.contracts import Comparison

_NUMERIC_TYPES = frozenset({"int2", "int4", "int8", "numeric", "float4", "float8"})


def numeric(value: SqlValue) -> Decimal | None:
    """Preserve decimal precision and reject booleans, NULL and nonfinite numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _cell(value: SqlValue, column_type: str) -> tuple[str, Decimal | str | None]:
    if value is None:
        return ("number" if column_type in _NUMERIC_TYPES else column_type, None)
    if column_type in _NUMERIC_TYPES:
        number = numeric(value)
        if number is not None:
            return ("number", number)
    return (column_type, str(value))


def _rows(result: QueryResultPayload) -> list[tuple[tuple[str, Decimal | str | None], ...]]:
    return [
        tuple(_cell(value, column.type) for value, column in zip(row, result.columns, strict=True))
        for row in result.rows
    ]


def compare(
    actual: QueryResultPayload,
    expected: QueryResultPayload,
    mode: Comparison,
    tolerance: float = 0.001,
) -> bool:
    """Ignore aliases only; duplicate rows, NULLs, types and truncation remain significant."""
    if actual.result_truncated or expected.result_truncated:
        return False
    if mode is Comparison.EMPTY:
        return actual.row_count == expected.row_count == 0
    if len(actual.columns) != len(expected.columns) or actual.row_count != expected.row_count:
        return False
    if mode is Comparison.SCALAR:
        return _scalar(actual, expected, tolerance)
    left_rows, right_rows = _rows(actual), _rows(expected)
    if mode is Comparison.ORDERED:
        return left_rows == right_rows
    return Counter(left_rows) == Counter(right_rows)


def _scalar(actual: QueryResultPayload, expected: QueryResultPayload, tolerance: float) -> bool:
    if actual.row_count != 1 or len(actual.columns) != 1:
        return False
    if any(r.columns[0].type not in _NUMERIC_TYPES for r in (actual, expected)):
        return False
    left, right = numeric(actual.rows[0][0]), numeric(expected.rows[0][0])
    if left is None or right is None:
        return actual.rows[0][0] is None and expected.rows[0][0] is None
    return abs(Fraction(left) - Fraction(right)) <= abs(Fraction(right)) * Fraction(str(tolerance))
