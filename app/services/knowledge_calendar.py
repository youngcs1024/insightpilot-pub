"""Pure calendar atoms for knowledge validity, sharing the business period resolver."""

import re
from datetime import date, datetime, timedelta

from app.core.errors import PeriodUnresolved
from app.schemas.retrieval import KnowledgeTimeScope, PointTimeScope, PolicyPeriod, RangeTimeScope
from app.services.periods import BUSINESS_TZ, Period, resolve_period

NUMBER = r"[0-9零〇一二三四五六七八九十两]+"  # noqa: RUF001 -- Chinese calendar zero.
YEAR = r"(?:[0-9]{4}|[零〇一二三四五六七八九]{4})"
DAY = rf"(?:(?:{YEAR})年)?{NUMBER}月{NUMBER}[日号]|[0-9]{{4}}-[0-9]{{1,2}}-[0-9]{{1,2}}"
MONTH = rf"(?:(?:{YEAR})年)?{NUMBER}月"
QUARTER = rf"(?:(?:{YEAR})年)?(?:Q[0-9]+|第{NUMBER}季度)"
RELATIVE = r"去年同期|年初至今|今天|今日|昨天|昨日|前天|明天|当前|目前|现在|上个月|上月|本月|下个月|下月|今年|去年|明年|(?:最近|过去)[+-]?[0-9]+(?:\.[0-9]+)?天"
_DIGITS = {
    char: value
    for value, chars in enumerate(("零〇", "一", "二两", "三", "四", "五", "六", "七", "八", "九"))
    for char in chars
}
_QUARTERS_IN_YEAR = 4


def number(value: str) -> int:
    """Read decimal digits or bounded Chinese calendar numerals, not general prose."""
    if value.isascii() and value.isdigit():
        return int(value)
    if "十" in value:
        tens, units = value.split("十")
        return (_DIGITS[tens] if tens else 1) * 10 + (_DIGITS[units] if units else 0)
    return int("".join(str(_DIGITS[char]) for char in value))


def explicit_year(expression: str) -> int | None:
    """Find the year attached to a complete calendar atom."""
    match = re.match(rf"({YEAR})(?:年|-)", expression)
    return number(match[1]) if match else None


def year_of(scope: KnowledgeTimeScope) -> int | None:
    """Only a single calendar year may fill a missing year in a follow-up."""
    if isinstance(scope, PointTimeScope):
        return scope.as_of.year
    years = {
        bound.year for item in scope.periods for bound in (item.start, item.end - timedelta(days=1))
    }
    return next(iter(years)) if len(years) == 1 else None


def period_scope(period: Period) -> PolicyPeriod:
    """Ceil partial ending days: policy validity is date-based, unlike business SQL."""
    end = period.end.date()
    if period.end.timetz().replace(tzinfo=None) != datetime.min.time():
        end += timedelta(days=1)
    return PolicyPeriod(start=period.start.date(), end=end, label=period.label)


def _month(year: int, month: int, label: str) -> PolicyPeriod:
    return period_scope(
        resolve_period(f"{year}年{month}月", now=datetime(year, 1, 1, tzinfo=BUSINESS_TZ))
    ).model_copy(update={"label": label})


def calendar_atom(expression: str, *, now: datetime, year: int) -> PolicyPeriod:
    """Resolve one complete day, month, quarter or year; reject impossible bounds."""
    if re.fullmatch(r"[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}", expression):
        parts = [int(part) for part in expression.split("-")]
        day = date(*parts)
        return PolicyPeriod(start=day, end=day + timedelta(days=1), label=expression)
    match = re.fullmatch(rf"(?:({YEAR})年)?({NUMBER})月(?:({NUMBER})[日号])?", expression)
    if match:
        actual_year = number(match[1]) if match[1] else year
        month = number(match[2])
        if match[3] is None:
            return _month(actual_year, month, f"{actual_year}年{month}月")
        day = date(actual_year, month, number(match[3]))
        return PolicyPeriod(start=day, end=day + timedelta(days=1), label=day.isoformat())
    match = re.fullmatch(rf"(?:({YEAR})年)?(?:Q([0-9]+)|第({NUMBER})季度)", expression)
    if match:
        actual_year = number(match[1]) if match[1] else year
        quarter = number(match[2] or match[3])
        if not 1 <= quarter <= _QUARTERS_IN_YEAR:
            raise PeriodUnresolved()
        local = now.replace(year=actual_year, month=1, day=1)
        return period_scope(resolve_period(f"Q{quarter}", now=local))
    if re.fullmatch(rf"{YEAR}年", expression):
        actual_year = number(expression[:-1])
        return PolicyPeriod(
            start=date(actual_year, 1, 1), end=date(actual_year + 1, 1, 1), label=expression
        )
    raise PeriodUnresolved()


def relative_atom(
    expression: str, *, now: datetime, reference: KnowledgeTimeScope | None
) -> list[PolicyPeriod]:
    """Relative words always use the request clock, except explicit last-year references."""
    offsets = {
        "今天": 0,
        "今日": 0,
        "当前": 0,
        "目前": 0,
        "现在": 0,
        "昨天": -1,
        "昨日": -1,
        "前天": -2,
        "明天": 1,
    }
    if expression in offsets:
        day = now.date() + timedelta(days=offsets[expression])
        return [PolicyPeriod(start=day, end=day + timedelta(days=1), label=expression)]
    if expression in {"去年", "明年"}:
        year = now.year + (-1 if expression == "去年" else 1)
        return [calendar_atom(f"{year}年", now=now, year=year)]
    if expression in {"下个月", "下月"}:
        next_month = date(now.year, now.month, 1) + timedelta(days=32)
        return [_month(next_month.year, next_month.month, expression)]
    if expression == "去年同期":
        if reference is None:
            raise PeriodUnresolved()
        items = scope_periods(reference)
        return [
            period_scope(
                resolve_period(
                    expression,
                    now=now,
                    reference_period=Period(
                        start=datetime.combine(item.start, datetime.min.time(), BUSINESS_TZ),
                        end=datetime.combine(item.end, datetime.min.time(), BUSINESS_TZ),
                        label=item.label or "上文时期",
                    ),
                )
            )
            for item in items
        ]
    return [period_scope(resolve_period("上个月" if expression == "上月" else expression, now=now))]


def scope_periods(scope: KnowledgeTimeScope) -> list[PolicyPeriod]:
    """Project point scopes to a day interval for semantic comparison only."""
    if isinstance(scope, PointTimeScope):
        return [PolicyPeriod(start=scope.as_of, end=scope.as_of + timedelta(days=1))]
    return scope.periods


def same_scope(left: KnowledgeTimeScope, right: KnowledgeTimeScope) -> bool:
    """Compare each requested period, ignoring labels but never merging comparisons."""

    def bounds(scope: KnowledgeTimeScope) -> list[tuple[date, date]]:
        return sorted((item.start, item.end) for item in scope_periods(scope))

    return bounds(left) == bounds(right)


def as_scope(periods: list[PolicyPeriod], *, point: bool) -> KnowledgeTimeScope:
    """Preserve labelled comparisons, including adjacent periods."""
    if point and len(periods) == 1:
        return PointTimeScope(as_of=periods[0].start)
    return RangeTimeScope(periods=periods)
