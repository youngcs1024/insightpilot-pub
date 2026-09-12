"""Deterministic Shanghai calendar intervals; no clock reads or external calls."""

import re
from calendar import monthrange
from datetime import datetime, timedelta
from typing import Self
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from app.core.errors import PeriodUnresolved
from app.schemas.mcp import Contract

BUSINESS_TZ = ZoneInfo("Asia/Shanghai")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_QUARTERS = {"第一季度": 1, "第二季度": 2, "第三季度": 3, "第四季度": 4}


def _shanghai(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PeriodUnresolved("A timezone-aware reference instant is required.")
    return value.astimezone(BUSINESS_TZ)


def _display(value: datetime) -> str:
    precision = "microseconds" if value.microsecond else "seconds" if value.second else "minutes"
    return value.replace(tzinfo=None).isoformat(sep=" ", timespec=precision)


class Period(Contract):
    """An immutable nonempty half-open business interval."""

    model_config = ConfigDict(frozen=True)

    start: AwareDatetime
    end: AwareDatetime
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_bounds(self) -> Self:
        """Normalize instants before checking interval order."""
        try:
            start, end = _shanghai(self.start), _shanghai(self.end)
        except (ValueError, OverflowError) as exc:
            raise PeriodUnresolved("Period boundaries exceed the supported calendar.") from exc
        if start >= end:
            raise PeriodUnresolved("Period start must precede its exclusive end.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        return self

    def as_assumption(self) -> str:
        """State exact bounds without hiding sub-minute precision."""
        return (
            f"{self.label} 定义为 {_display(self.start)} 至 {_display(self.end)}"
            "（Asia/Shanghai，左闭右开）"
        )


def _month(year: int, month: int, *, months: int = 1, label: str) -> Period:
    start = datetime(year, month, 1, tzinfo=BUSINESS_TZ)
    end_year, end_month = divmod(year * 12 + month - 1 + months, 12)
    return Period(
        start=start, end=datetime(end_year, end_month + 1, 1, tzinfo=BUSINESS_TZ), label=label
    )


def _previous_year(value: datetime) -> datetime:
    year = value.year - 1
    day = min(value.day, monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _relative(expr: str, now: datetime, reference: Period | None) -> Period | None:
    if expr == "去年同期":
        if reference is None:
            raise PeriodUnresolved("Last year's equivalent requires a reference period.")
        return Period(
            start=_previous_year(reference.start),
            end=_previous_year(reference.end),
            label=f"{reference.label}的去年同期",
        )
    if expr == "年初至今":
        return Period(start=datetime(now.year, 1, 1, tzinfo=BUSINESS_TZ), end=now, label=expr)
    if expr == "今年":
        return _month(now.year, 1, months=12, label=f"{now.year}年")
    if expr in {"本月", "上个月"}:
        year, month = now.year, now.month
        if expr == "上个月":
            year, month = divmod(year * 12 + month - 2, 12)
            month += 1
        return _month(year, month, label=f"{year}年{month}月")
    if match := re.fullmatch(r"(?:最近|过去)([0-9]+)天", expr):
        days = int(match[1])
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return Period(start=end - timedelta(days=days), end=end, label=expr)
    return None


def _calendar_expression(expr: str, now: datetime) -> Period:
    if match := re.fullmatch(r"(?:([0-9]{4})年)?([0-9]{1,2})月", expr):
        year, month = int(match[1]) if match[1] else now.year, int(match[2])
        return _month(year, month, label=f"{year}年{month}月")
    quarter_match = re.fullmatch(r"Q([1-4])", expr)
    quarter = int(quarter_match[1]) if quarter_match else _QUARTERS.get(expr)
    if quarter is not None:
        return _month(now.year, (quarter - 1) * 3 + 1, months=3, label=f"{now.year}年Q{quarter}")
    if match := re.fullmatch(
        r"([0-9]{4}-[0-9]{2}-[0-9]{2})\s*到\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", expr
    ):
        start = datetime.fromisoformat(match[1]).replace(tzinfo=BUSINESS_TZ)
        end = datetime.fromisoformat(match[2]).replace(tzinfo=BUSINESS_TZ) + timedelta(days=1)
        return Period(start=start, end=end, label=f"{match[1]} 到 {match[2]}")
    raise PeriodUnresolved("Unsupported complete period expression.")


def resolve_period(expr: str, *, now: datetime, reference_period: Period | None = None) -> Period:
    """Resolve a complete expression using an explicit aware request instant.

    Recent days exclude today. Explicit date ranges include their final date.
    Last-year comparison clamps February 29 to February 28 independently at
    each boundary; a collapsed interval requires clarification.
    """
    try:
        local_now = _shanghai(now)
        expression = expr.strip()
        relative = _relative(expression, local_now, reference_period)
        return relative if relative is not None else _calendar_expression(expression, local_now)
    except (ValueError, OverflowError) as exc:
        raise PeriodUnresolved("Invalid date or period outside the supported calendar.") from exc


def build_date_context(*, now: datetime) -> str:
    """Render stable date context independent of the host clock and locale."""
    try:
        today = _shanghai(now)
    except (ValueError, OverflowError) as exc:
        raise PeriodUnresolved("Reference instant exceeds the supported calendar.") from exc
    iso = today.isocalendar()
    return (
        f"Current date: {today.date().isoformat()} ({_WEEKDAYS[today.weekday()]})\n"
        f"Timezone: Asia/Shanghai\nCurrent year: {today.year}, "
        f"Quarter: Q{(today.month - 1) // 3 + 1}, ISO week: {iso.year}-W{iso.week:02d}\n"
        "Interpret relative expressions against this request instant: "
        "上个月, 本月, 今年, 年初至今, 最近N天, 过去N天. "
        "Recent N days exclude today; 去年同期 requires an explicit reference period.\n"
    )
