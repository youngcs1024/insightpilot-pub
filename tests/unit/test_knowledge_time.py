"""Calendar extraction is offline and never substitutes for live retrieval quality."""

from datetime import UTC, date, datetime

import pytest

from app.core.errors import PeriodUnresolved
from app.schemas.retrieval import PointTimeScope, PolicyPeriod, RangeTimeScope
from app.services.knowledge_calendar import same_scope
from app.services.knowledge_time import needs_history, parse_time
from app.services.periods import BUSINESS_TZ

NOW = datetime(2026, 9, 8, 12, tzinfo=BUSINESS_TZ)


@pytest.mark.parametrize(
    ("question", "day"),
    [
        ("2024年2月29日的政策", "2024-02-29"),
        ("二〇二六年八月十五号政策", "2026-08-15"),
        ("截至2026-08-01适用的规则", "2026-08-01"),
        ("今天政策", "2026-09-08"),
        ("今日政策", "2026-09-08"),
        ("目前政策", "2026-09-08"),
        ("当前政策", "2026-09-08"),
        ("现在政策", "2026-09-08"),
        ("昨天政策", "2026-09-07"),
        ("昨日政策", "2026-09-07"),
        ("前天政策", "2026-09-06"),
        ("明天政策", "2026-09-09"),
    ],
)
def test_point_dates(question: str, day: str) -> None:
    result = parse_time(question, now=NOW)
    assert result.clarification is None
    assert result.scope == PointTimeScope(as_of=date.fromisoformat(day))


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("2026年8月政策", "2026-08-01", "2026-09-01"),
        ("2026年政策", "2026-01-01", "2027-01-01"),
        ("2025年Q4政策", "2025-10-01", "2026-01-01"),
        ("2026年第二季度政策", "2026-04-01", "2026-07-01"),
        ("Q3规则", "2026-07-01", "2026-10-01"),
        ("今年政策", "2026-01-01", "2027-01-01"),
        ("去年政策", "2025-01-01", "2026-01-01"),
        ("明年政策", "2027-01-01", "2028-01-01"),
        ("本月规则", "2026-09-01", "2026-10-01"),
        ("上个月规则", "2026-08-01", "2026-09-01"),
        ("上月规则", "2026-08-01", "2026-09-01"),
        ("下个月规则", "2026-10-01", "2026-11-01"),
        ("下月规则", "2026-10-01", "2026-11-01"),
        ("最近7天规则", "2026-09-01", "2026-09-08"),
        ("过去30天规则", "2026-08-09", "2026-09-08"),
        ("年初至今规则", "2026-01-01", "2026-09-09"),
        ("2026-07-01 到 2026-08-31 政策", "2026-07-01", "2026-09-01"),
        ("2026年7月1日至2026年7月31日政策", "2026-07-01", "2026-08-01"),
        ("2026年7月至8月政策", "2026-07-01", "2026-09-01"),
    ],
)
def test_half_open_ranges(question: str, start: str, end: str) -> None:
    result = parse_time(question, now=NOW)
    assert result.clarification is None
    assert isinstance(result.scope, RangeTimeScope)
    assert [(item.start.isoformat(), item.end.isoformat()) for item in result.scope.periods] == [
        (start, end)
    ]


@pytest.mark.parametrize(
    "question",
    [
        "2026-02-29政策",
        "2026年13月政策",
        "2026年8月32日政策",
        "2026年7月和13月政策",
        "2026年7月和2026/08/01政策",
        "2026年8月上旬政策",
        "2026年8月至7月政策",
        "2026-08-31到2026-08-01政策",
        "最近0天政策",
        "过去-1天政策",
        "最近1.5天政策",
        "Q5政策",
        "第五季度政策",
        "2026年-1月政策",
        "近期政策",
        "2026年春节政策",
        "去年同期政策",
        "9999年12月政策",
        "0000年政策",
        "2026年8月和2025年9月以及10月政策",
        "9999-12-31政策",
        "2026年8月初政策",
        "2026年8月底政策",
        "2026年8月以前的退货政策",
        "2026年8月以来的退货政策",
        "2026年7、8月政策",
        "今天（2026-08-01）的政策",
        "2026年8月1日12时的政策",
    ],
)
def test_unresolved_explicit_time_never_defaults(question: str) -> None:
    result = parse_time(question, now=NOW)
    assert result.clarification is not None
    assert result.clarification.kind.value == "period_unresolved"
    assert result.scope is None


@pytest.mark.parametrize(
    "question",
    [
        "七天无理由退货政策",
        "30天售后政策",
        "SKU-A1023政策",
        "产品SKU-2026-08-01的规则",
        "收到货以后七天内申请退货",
    ],
)
def test_durations_and_skus_are_not_query_dates(question: str) -> None:
    result = parse_time(question, now=NOW)
    assert result.clarification is None
    assert result.scope is None


def test_multiple_periods_retain_labels_and_shared_explicit_year() -> None:
    result = parse_time("比较2025年七月和八月政策", now=NOW)
    assert [item.label for item in result.scope.periods] == ["2025年7月", "2025年8月"]
    assert result.assumptions == []
    assert result.scope.periods[0].end == result.scope.periods[1].start


def test_missing_year_is_disclosed_and_shanghai_not_host_date() -> None:
    result = parse_time("1月政策", now=datetime(2026, 12, 31, 16, 30, tzinfo=UTC))
    assert result.scope.periods[0].start == date(2027, 1, 1)
    assert "2027" in result.assumptions[0]
    assert "Asia/Shanghai" in result.assumptions[0]
    assert result.year_inferred


def test_explicit_reference_year_and_last_year_equivalent() -> None:
    result = parse_time("8月政策", now=NOW, reference_year=2024)
    assert result.scope.periods[0].start == date(2024, 8, 1)
    assert "上文" in result.assumptions[0]
    reference = RangeTimeScope(
        periods=[PolicyPeriod(start=date(2024, 2, 1), end=date(2024, 3, 1), label="闰年二月")]
    )
    previous = parse_time("去年同期政策", now=NOW, reference=reference)
    assert previous.scope.periods[0].start == date(2023, 2, 1)
    assert previous.scope.periods[0].end == date(2023, 3, 1)


def test_naive_clock_is_rejected() -> None:
    with pytest.raises(PeriodUnresolved):
        parse_time("当前政策", now=datetime(2026, 9, 8))


def test_too_many_periods_clarifies() -> None:
    assert parse_time("和".join(["2026年8月"] * 25), now=NOW).clarification


@pytest.mark.parametrize("numeral", ["1", "一", "零"])
def test_long_numeral_cannot_be_partially_accepted(numeral: str) -> None:
    result = parse_time(numeral * 20_000 + "一月的政策", now=NOW)
    assert result.clarification is not None
    assert result.scope is None


def test_long_numeric_context_does_not_hide_an_explicit_calendar_date() -> None:
    result = parse_time("1" * 20_000 + "问2026年8月政策", now=NOW)
    assert result.clarification is None
    assert result.scope.periods[0].start == date(2026, 8, 1)


def test_scope_comparison_preserves_comparison_structure() -> None:
    two = parse_time("2026年7月和8月", now=NOW).scope
    joined = parse_time("2026年7月至8月", now=NOW).scope
    assert not same_scope(two, joined)
    assert same_scope(
        parse_time("2026-08-01", now=NOW).scope, parse_time("2026年8月1日", now=NOW).scope
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("那运费呢\uff1f", True),
        ("它有什么限制", True),
        ("当时的运费", True),
        ("What about shipping?", True),
        ("2026年8月退款政策是什么\uff1f", False),
        ("七天无理由退货政策", False),
    ],
)
def test_followup_guard(question: str, expected: bool) -> None:
    assert needs_history(question) is expected
