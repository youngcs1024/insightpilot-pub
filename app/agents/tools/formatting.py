"""Presentation helpers; evidence retains its exact original numeric values."""

from decimal import Decimal, ROUND_HALF_UP, localcontext

from app.agents.tools.arithmetic import Number, finite_decimal

_TWO_PLACES = Decimal("0.01")
_WAN = Decimal(10_000)
_YI = Decimal(100_000_000)


def _rounded(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + 4)
        return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def format_currency(value: Number) -> str:
    """Render a CNY amount with grouping and two display-only decimal places."""
    return f"¥{_rounded(finite_decimal(value)):,.2f}"


def format_percent(value: Number) -> str:
    """Render a value already expressed in percentage units."""
    return f"{_rounded(finite_decimal(value)):.2f}%"


def format_large_number(value: Number) -> str:
    """Use 万 or 亿 at the corresponding absolute-value threshold."""
    number = finite_decimal(value)
    magnitude = abs(number)
    if magnitude >= _YI:
        return f"{_rounded(number / _YI):.2f}亿"
    if magnitude >= _WAN:
        return f"{_rounded(number / _WAN):.2f}万"
    return f"{_rounded(number):,.2f}"
