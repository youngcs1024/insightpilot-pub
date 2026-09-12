"""Calendar semantics and failure contracts without a wall clock or external I/O."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.errors import PeriodUnresolved, ValidationError
from app.services.periods import BUSINESS_TZ, Period, build_date_context, resolve_period

NOW = datetime(2026, 9, 8, 14, 23, 45, 123456, tzinfo=BUSINESS_TZ)


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=BUSINESS_TZ)


@pytest.mark.parametrize(
    ("expr", "start", "end"),
    [
        ("2026年8月", "2026-08-01", "2026-09-01"),
        (" 8月 \n", "2026-08-01", "2026-09-01"),
        ("2026年12月", "2026-12-01", "2027-01-01"),
        ("2024年2月", "2024-02-01", "2024-03-01"),
        ("2025年2月", "2025-02-01", "2025-03-01"),
    ],
)
def test_month_expression_is_half_open(expr: str, start: str, end: str) -> None:
    period = resolve_period(expr, now=NOW)
    assert period.start == instant(start)
    assert period.end == instant(end)
    assert period.start.tzinfo == period.end.tzinfo == BUSINESS_TZ


def test_month_boundary_is_shanghai_not_utc() -> None:
    period = resolve_period("2026年8月", now=NOW)
    assert period.start.astimezone(UTC) == datetime(2026, 7, 31, 16, tzinfo=UTC)
    assert period.end.astimezone(UTC) == datetime(2026, 8, 31, 16, tzinfo=UTC)


@pytest.mark.parametrize(
    ("expr", "start", "end"),
    [
        ("上个月", "2026-08-01", "2026-09-01"),
        ("本月", "2026-09-01", "2026-10-01"),
        ("今年", "2026-01-01", "2027-01-01"),
        ("最近7天", "2026-09-01", "2026-09-08"),
        ("过去30天", "2026-08-09", "2026-09-08"),
        ("最近1天", "2026-09-07", "2026-09-08"),
        ("过去1天", "2026-09-07", "2026-09-08"),
    ],
)
def test_relative_expressions_resolve_against_now(expr: str, start: str, end: str) -> None:
    period = resolve_period(expr, now=NOW)
    assert (period.start, period.end) == (instant(start), instant(end))


@pytest.mark.parametrize(
    ("expr", "start", "end"),
    [
        ("Q1", "2026-01-01", "2026-04-01"),
        ("第一季度", "2026-01-01", "2026-04-01"),
        ("Q2", "2026-04-01", "2026-07-01"),
        ("第二季度", "2026-04-01", "2026-07-01"),
        ("Q3", "2026-07-01", "2026-10-01"),
        ("第三季度", "2026-07-01", "2026-10-01"),
        ("Q4", "2026-10-01", "2027-01-01"),
        ("第四季度", "2026-10-01", "2027-01-01"),
    ],
)
def test_quarter_resolution(expr: str, start: str, end: str) -> None:
    period = resolve_period(expr, now=NOW)
    assert (period.start, period.end) == (instant(start), instant(end))


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "昨天那个时间",
        "查询8月销售额",
        "8月和9月",
        "Q0",
        "Q5",
        "第五季度",
        "0月",
        "13月",
        "0000年1月",
        "最近0天",
        "过去0天",
        "过去-1天",
        "最近1.5天",
        "2026-02-29 到 2026-03-01",
        "2026-08-15 到 2026-08-14",
        "9999年12月",
        "9999-12-31 到 9999-12-31",
        "最近999999999999天",
        "去年同期",
    ],
)
def test_unparseable_raises(expr: str) -> None:
    with pytest.raises(PeriodUnresolved) as caught:
        resolve_period(expr, now=NOW)
    assert isinstance(caught.value, ValidationError)
    assert caught.value.code == "PERIOD_UNRESOLVED"
    assert not caught.value.retryable


def test_assumption_string_states_timezone_and_bounds() -> None:
    assert resolve_period("2026年8月", now=NOW).as_assumption() == (
        "2026年8月 定义为 2026-08-01 00:00 至 2026-09-01 00:00（Asia/Shanghai，左闭右开）"
    )


@pytest.mark.parametrize("expr", ["2026年1月", "2026年7月"])
def test_dst_not_applicable_for_shanghai(expr: str) -> None:
    # Modern business dates only: Shanghai did historically observe DST.
    period = resolve_period(expr, now=NOW)
    assert period.start.utcoffset() == period.end.utcoffset() == timedelta(hours=8)
    assert period.start.dst() == period.end.dst() == timedelta(0)


def test_utc_reference_uses_shanghai_year_and_month() -> None:
    now = datetime(2026, 12, 31, 16, 30, tzinfo=UTC)
    assert resolve_period("1月", now=now).start == instant("2027-01-01")
    assert resolve_period("上个月", now=now).start == instant("2026-12-01")
    assert resolve_period("Q1", now=now).end == instant("2027-04-01")
    assert resolve_period("最近1天", now=now).end == instant("2027-01-01")


def test_recent_days_include_leap_day_and_cross_year() -> None:
    leap = resolve_period("最近2天", now=instant("2024-03-01"))
    assert (leap.start, leap.end) == (instant("2024-02-28"), instant("2024-03-01"))
    year = resolve_period("过去7天", now=instant("2026-01-03"))
    assert year.start == instant("2025-12-27")


@pytest.mark.parametrize(
    ("expr", "start", "end"),
    [
        ("2026-08-01 到 2026-08-15", "2026-08-01", "2026-08-16"),
        ("2026-08-15到2026-08-15", "2026-08-15", "2026-08-16"),
        ("2026-12-31 到 2026-12-31", "2026-12-31", "2027-01-01"),
        ("2024-02-28 到 2024-02-29", "2024-02-28", "2024-03-01"),
    ],
)
def test_date_range_includes_entire_final_date(expr: str, start: str, end: str) -> None:
    period = resolve_period(expr, now=NOW)
    assert (period.start, period.end) == (instant(start), instant(end))


@pytest.mark.parametrize("now", [NOW, NOW.replace(microsecond=0)])
def test_year_to_date_preserves_exact_request_instant(now: datetime) -> None:
    period = resolve_period("年初至今", now=now.astimezone(UTC))
    assert period.start == instant("2026-01-01")
    assert period.end == now
    assert now.replace(tzinfo=None).isoformat(sep=" ") in period.as_assumption()


def test_empty_year_to_date_requires_clarification() -> None:
    with pytest.raises(PeriodUnresolved):
        resolve_period("年初至今", now=instant("2026-01-01"))


@pytest.mark.parametrize(
    ("start", "end", "expected_start", "expected_end"),
    [
        ("2024-02-29", "2024-03-01", "2023-02-28", "2023-03-01"),
        ("2024-02-01", "2024-02-29", "2023-02-01", "2023-02-28"),
        ("2025-12-01", "2026-01-01", "2024-12-01", "2025-01-01"),
        (
            "2024-02-29T12:34:56.123456",
            "2024-03-01T12:34:56.123456",
            "2023-02-28T12:34:56.123456",
            "2023-03-01T12:34:56.123456",
        ),
    ],
)
def test_last_year_maps_reference_boundaries(
    start: str, end: str, expected_start: str, expected_end: str
) -> None:
    reference = Period(start=instant(start), end=instant(end), label="比较区间")
    period = resolve_period("去年同期", now=NOW, reference_period=reference)
    assert (period.start, period.end) == (instant(expected_start), instant(expected_end))
    assert period.label == "比较区间的去年同期"


def test_collapsed_leap_mapping_requires_clarification() -> None:
    reference = Period(start=instant("2024-02-28"), end=instant("2024-02-29"), label="闰日前")
    with pytest.raises(PeriodUnresolved):
        resolve_period("去年同期", now=NOW, reference_period=reference)


@pytest.mark.parametrize("expr", ["2026年8月", "上个月"])
def test_naive_request_instant_is_rejected_even_for_absolute_expression(expr: str) -> None:
    with pytest.raises(PeriodUnresolved):
        resolve_period(expr, now=NOW.replace(tzinfo=None))


def test_calendar_underflow_is_typed() -> None:
    reference = Period(start=instant("0001-01-01"), end=instant("0001-02-01"), label="最早年份")
    with pytest.raises(PeriodUnresolved):
        resolve_period("去年同期", now=NOW, reference_period=reference)
    with pytest.raises(PeriodUnresolved):
        resolve_period("上个月", now=reference.start)
    with pytest.raises(PeriodUnresolved):
        resolve_period("最近1天", now=reference.start)


def test_period_is_immutable_and_normalizes_aware_bounds() -> None:
    period = Period(
        start=datetime(2026, 7, 31, 16, tzinfo=UTC),
        end=datetime(2026, 8, 31, 16, tzinfo=UTC),
        label="8月",
    )
    assert period.start == instant("2026-08-01")
    assert period.start.tzinfo == period.end.tzinfo == BUSINESS_TZ
    with pytest.raises(PydanticValidationError):
        period.label = "changed"
    assert Period.model_validate_json(period.model_dump_json()) == period


@pytest.mark.parametrize("end", ["2026-08-01", "2026-07-31"])
def test_period_rejects_empty_or_reversed_bounds(end: str) -> None:
    with pytest.raises(PeriodUnresolved):
        Period(start=instant("2026-08-01"), end=instant(end), label="invalid")


def test_period_rejects_naive_boundary() -> None:
    with pytest.raises(PydanticValidationError):
        Period(start=NOW.replace(tzinfo=None), end=NOW, label="invalid")


def test_date_context_uses_local_date_and_iso_week_year() -> None:
    context = build_date_context(now=datetime(2020, 12, 31, 16, tzinfo=UTC))
    assert "Current date: 2021-01-01 (Friday)" in context
    assert "Timezone: Asia/Shanghai" in context
    assert "Current year: 2021, Quarter: Q1, ISO week: 2020-W53" in context
    assert "去年同期 requires an explicit reference period" in context
    assert context == build_date_context(now=instant("2021-01-01"))


def test_date_context_rejects_naive_reference() -> None:
    with pytest.raises(PeriodUnresolved):
        build_date_context(now=NOW.replace(tzinfo=None))


def test_timezone_conversion_overflow_is_typed() -> None:
    now = datetime.max.replace(tzinfo=UTC)
    with pytest.raises(PeriodUnresolved):
        resolve_period("本月", now=now)
    with pytest.raises(PeriodUnresolved):
        build_date_context(now=now)
    with pytest.raises(PeriodUnresolved):
        Period(start=NOW, end=now, label="overflow")
