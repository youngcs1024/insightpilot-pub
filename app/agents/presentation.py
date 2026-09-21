"""Explicit presentation requests never change analytical context or evidence."""

import re

from app.schemas.memory import FormatPreferenceContent

_STYLE = re.compile(
    r"(?P<prose>不要(?:用)?表格|不用表格|(?:用|以)(?:文字|段落|正文)(?:回答|说明|呈现)?|"
    r"(?:in|as|use)\s+(?:plain\s+)?prose|no\s+tables?)|"
    r"(?P<table>(?:用|以|使用)表格|列成表格|(?:in|as|use)\s+(?:a\s+)?table)",
    re.IGNORECASE,
)
_DECIMALS = re.compile(
    r"(?:保留|精确到)\s*([0-4零一二两三四])\s*位小数|\b([0-4])\s+decimal\s+places?\b",
    re.IGNORECASE,
)
_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4}


def resolve_preference(
    question: str, saved: FormatPreferenceContent | None
) -> FormatPreferenceContent | None:
    """Overlay explicit fields only; the last explicit instruction wins per field."""
    styles = list(_STYLE.finditer(question))
    decimals = list(_DECIMALS.finditer(question))
    if not styles and not decimals:
        return saved.model_copy(deep=True) if saved else None
    prefer = saved.prefer if saved else "prose"
    places = saved.decimals if saved else 2
    if styles:
        prefer = "table" if styles[-1].group("table") else "prose"
    if decimals:
        value = decimals[-1].group(1) or decimals[-1].group(2)
        places = _DIGITS[value] if value in _DIGITS else int(value)
    return FormatPreferenceContent(prefer=prefer, decimals=places)
