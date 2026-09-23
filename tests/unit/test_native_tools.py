"""Pure native calculations, formatting and argument boundaries."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.agents.tools.arithmetic import growth_rate, percentage_change, share_of_total
from app.agents.tools.formatting import format_currency, format_large_number, format_percent
from app.agents.tools.native import definitions, invoke
from app.agents.tools.periods import resolve_period
from app.core.errors import NativeArithmeticError, NativeToolError, PeriodUnresolved

NOW = datetime(2026, 9, 8, tzinfo=UTC)


def test_percentage_change_handles_zero_base() -> None:
    with pytest.raises(NativeArithmeticError) as caught:
        percentage_change("0", "20")
    assert caught.value.code == "NATIVE_ARITHMETIC_UNDEFINED"


def test_growth_rate_handles_negative() -> None:
    assert growth_rate("-20", "-10") == Decimal("50")
    assert percentage_change("20", "25") == Decimal("25")


def test_share_and_format_percent_use_percentage_units() -> None:
    assert share_of_total("1", "8") == Decimal("12.500")
    assert format_percent(share_of_total("1", "8")) == "12.50%"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "broken", "1e1001", True])
def test_invalid_numbers_fail_typed(value: str | bool) -> None:
    with pytest.raises(NativeArithmeticError):
        percentage_change(value, "1")


def test_share_zero_total_fails_typed() -> None:
    with pytest.raises(NativeArithmeticError):
        share_of_total("1", "0")


def test_format_currency_cny() -> None:
    assert format_currency("1234.5") == "¥1,234.50"
    assert format_currency("0.005") == "¥0.01"


def test_format_large_number_uses_chinese_units() -> None:
    assert format_large_number("10000") == "1.00万"
    assert format_large_number("123456789") == "1.23亿"
    assert format_large_number("-10000") == "-1.00万"


def test_native_period_uses_injected_instant_and_shanghai_timezone() -> None:
    period = resolve_period("上个月", now=NOW)
    assert period.start.isoformat() == "2026-08-01T00:00:00+08:00"
    assert period.end.isoformat() == "2026-09-01T00:00:00+08:00"
    with pytest.raises(PeriodUnresolved):
        resolve_period("去年同期", now=NOW)


def test_native_tool_arguments_are_validated() -> None:
    assert [item.name for item in definitions(["arithmetic"])] == ["calculate_percentage"]
    assert (
        invoke(
            "calculate_percentage",
            '{"operation":"percentage_change","first":"10","second":"12"}',
            kinds=["arithmetic"],
            now=NOW,
            reference_period=None,
        )
        == "20.0"
    )
    with pytest.raises(PydanticValidationError):
        invoke(
            "calculate_percentage",
            '{"operation":"percentage_change","first":"10"}',
            kinds=["arithmetic"],
            now=NOW,
            reference_period=None,
        )
    with pytest.raises(NativeToolError):
        invoke(
            "resolve_period",
            '{"expression":"上个月"}',
            kinds=["arithmetic"],
            now=NOW,
            reference_period=None,
        )
