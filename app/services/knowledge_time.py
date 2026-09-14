"""Deterministic time extraction from questions; unmatched explicit dates never default."""

import re
import unicodedata
from datetime import datetime

from pydantic import ValidationError

from app.core.errors import PeriodUnresolved
from app.schemas.knowledge_query import (
    KnowledgeClarification,
    KnowledgeClarificationKind,
    KnowledgeTimeResolution,
)
from app.schemas.retrieval import KnowledgeTimeScope, PolicyPeriod
from app.services.knowledge_calendar import (
    DAY,
    MONTH,
    QUARTER,
    RELATIVE,
    YEAR,
    as_scope,
    calendar_atom,
    explicit_year,
    relative_atom,
)
from app.services.periods import BUSINESS_TZ

_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9_零\u3007一二三四五六七八九十两-])(?:(?P<day>{DAY})|(?P<quarter>{QUARTER})|"
    rf"(?P<month>{MONTH})|(?P<year>{YEAR}年)|(?P<relative>{RELATIVE}))"
)
_UNRESOLVED = re.compile(
    r"[0-9]{2,4}[-/][0-9]{1,8}(?:[-/][0-9]{1,8})?|[+-]?[0-9]{1,8}(?:\.[0-9]{1,8})?[年月日号]|"
    r"[零〇一二三四五六七八九十两]{1,8}[年月日号]|Q[0-9]+|季度|"
    r"上旬|中旬|下旬|月底|月初|年中|年底|年末|年初|最近|近期|过去|"
    r"(?:之前|之后|以前|以后)(?:的)?(?:政策|规则)|"
    r"上周|本周|下周|这周|春节|双十一|双11|去年|前年|明年"
)
_REFERENCE = re.compile(
    r"^(?:那|那么|还有|再说|再看)|(?:它|他们|它们|这个|那个|这些|那些|上述|前述|上面|"
    r"前面|刚才|之前提到|当时|同期|同上|同样|两者|二者)|"
    r"\b(?:it|those|that|them|same|what about|how about)\b",
    re.IGNORECASE,
)
_JOIN = re.compile(r"(?:到|至|~|—|-)")
_POINT_WORDS = {"今天", "今日", "当前", "目前", "现在", "昨天", "昨日", "前天", "明天"}
_MAX_PERIODS = 24


def needs_history(question: str) -> bool:
    """A conservative referential guard, independent from time and model failures."""
    text = question.strip()
    if _REFERENCE.search(text) is not None:
        return True
    return (
        re.search(r"(?:呢|怎么样)[?\uff1f。\uff01!]*$", text) is not None
        and re.search(r"什么|如何|哪|是否|多少|怎么(?!样)", text) is None
    )


def time_clarification() -> KnowledgeClarification:
    """Offer an actionable, fixed-format date request rather than exception prose."""
    return KnowledgeClarification(
        kind=KnowledgeClarificationKind.PERIOD_UNRESOLVED,
        message="请明确政策适用日期或各比较时期，例如 2026-08-01 或 2026年7月和8月；当前时间信息无法一致确定。",
    )


def reference_clarification() -> KnowledgeClarification:
    """Ambiguous topics are not treated as a failed retrieval or no evidence."""
    return KnowledgeClarification(
        kind=KnowledgeClarificationKind.REFERENCE_UNRESOLVED,
        message="请明确所指的知识主题及适用时间，例如“2026年8月退货政策中的运费规则”。",
    )


def _clean(question: str) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question))
    # Product identifiers and durations such as 七天/30天 are not calendar expressions.
    return re.sub(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9_-]+", " ", text)


def _check_remainder(text: str, matches: list[re.Match[str]]) -> None:
    remaining = list(text)
    for match in matches:
        suffix = text[match.end() :]
        if re.match(
            r"初|底|末|上旬|中旬|下旬|之前|之后|以前|以后|以来|起|前|后|[0-9]+(?:时|点|:)", suffix
        ):
            raise PeriodUnresolved()
        if re.search(r"[0-9]{1,8}[、和与/]+$", text[: match.start()]):
            raise PeriodUnresolved()
        remaining[match.start() : match.end()] = " " * (match.end() - match.start())
    if _UNRESOLVED.search("".join(remaining)):
        raise PeriodUnresolved()


def _duplicate_atom(previous: list[PolicyPeriod], current: list[PolicyPeriod], gap: str) -> bool:
    if not previous or re.fullmatch(r"[(),:即]*", gap) is None:
        return False
    if len(current) != 1:
        raise PeriodUnresolved()
    left, right = previous[-1], current[0]
    if (left.start, left.end) != (right.start, right.end):
        raise PeriodUnresolved()
    return True


def _periods(
    text: str,
    matches: list[re.Match[str]],
    now: datetime,
    reference: KnowledgeTimeScope | None,
    year: int,
) -> list[PolicyPeriod]:
    result: list[PolicyPeriod] = []
    previous_end = 0
    for match in matches:
        atom = match[0]
        current = (
            relative_atom(atom, now=now, reference=reference)
            if match.lastgroup == "relative"
            else [calendar_atom(atom, now=now, year=year)]
        )
        gap = text[previous_end : match.start()]
        if _duplicate_atom(result, current, gap):
            previous_end = match.end()
            continue
        if result and _JOIN.fullmatch(gap):
            if len(current) != 1 or result[-1].start >= current[0].end:
                raise PeriodUnresolved()
            previous = result.pop()
            result.append(
                PolicyPeriod(
                    start=previous.start,
                    end=current[0].end,
                    label=f"{previous.label}至{current[0].label}",
                )
            )
        else:
            result.extend(current)
        previous_end = match.end()
    return result


def parse_time(
    question: str,
    *,
    now: datetime,
    reference: KnowledgeTimeScope | None = None,
    reference_year: int | None = None,
) -> KnowledgeTimeResolution:
    """Extract all calendar atoms, retaining explicit ambiguity and comparison labels.

    The absence of any time is distinct from an unresolved explicit time. Default
    dates belong to the caller after necessary history resolution, never this parser.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise PeriodUnresolved()
    local_now = now.astimezone(BUSINESS_TZ)
    text = _clean(question)
    try:
        matches = list(_TOKEN.finditer(text))
        if len(matches) > _MAX_PERIODS:
            raise PeriodUnresolved()
        _check_remainder(text, matches)
        if not matches:
            return KnowledgeTimeResolution()
        years = {value for match in matches if (value := explicit_year(match[0])) is not None}
        yearless = any(
            match.lastgroup in {"day", "month", "quarter"} and explicit_year(match[0]) is None
            for match in matches
        )
        if yearless and len(years) > 1:
            raise PeriodUnresolved()
        year = next(iter(years)) if len(years) == 1 else reference_year or local_now.year
        periods = _periods(text, matches, local_now, reference, year)
        point = len(matches) == 1 and (
            matches[0].lastgroup == "day" or matches[0][0] in _POINT_WORDS
        )
        assumptions = []
        if yearless and not years:
            source = "上文" if reference_year is not None else "Asia/Shanghai 当前业务年"
            assumptions.append(f"未指定年份，按{source} {year} 年解释。")
        return KnowledgeTimeResolution(
            scope=as_scope(periods, point=point),
            assumptions=assumptions,
            year_inferred=yearless and not years,
        )
    except (ValueError, KeyError, OverflowError, ValidationError, PeriodUnresolved):
        return KnowledgeTimeResolution(clarification=time_clarification())
