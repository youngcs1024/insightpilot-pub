"""Finite, deterministic percentage calculations with an explicit unit contract."""

from decimal import Decimal, InvalidOperation

from app.core.errors import NativeArithmeticError

type Number = Decimal | int | str
MAX_NUMERIC_CHARS = 100
MAX_EXPONENT = 1000


def finite_decimal(value: Number) -> Decimal:
    """Reject booleans, malformed values and non-finite numbers at the native boundary."""
    if isinstance(value, bool):
        raise NativeArithmeticError("Boolean is not a numeric input.")
    raw = str(value)
    if len(raw) > MAX_NUMERIC_CHARS:
        raise NativeArithmeticError("Numeric input exceeds the supported size.")
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise NativeArithmeticError("Invalid numeric input.") from exc
    if not number.is_finite():
        raise NativeArithmeticError("Non-finite numeric input.")
    if number and abs(number.adjusted()) > MAX_EXPONENT:
        raise NativeArithmeticError("Numeric input exceeds the supported magnitude.")
    return number


def percentage_change(baseline: Number, current: Number) -> Decimal:
    """Return percentage units: (current - baseline) / abs(baseline) * 100."""
    old, new = finite_decimal(baseline), finite_decimal(current)
    if old == 0:
        raise NativeArithmeticError("Zero baseline.")
    return (new - old) / abs(old) * 100


def growth_rate(baseline: Number, current: Number) -> Decimal:
    """Return the same percentage-unit change for a business growth comparison."""
    return percentage_change(baseline, current)


def share_of_total(part: Number, total: Number) -> Decimal:
    """Return part / total * 100 in percentage units."""
    numerator, denominator = finite_decimal(part), finite_decimal(total)
    if denominator == 0:
        raise NativeArithmeticError("Zero total.")
    return numerator / denominator * 100
