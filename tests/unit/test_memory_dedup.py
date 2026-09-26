"""Type-specific matching, including deterministic Chinese comparison boundaries."""

# ruff: noqa: PLR2004 -- fixed threshold, row-count and pagination acceptance values.

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.config_models import RateLimitSettings, RateRule
from app.schemas.memory_write import WriteOutcome, WriteStatus
from app.services.memory.dedup import MemoryMatch, compare, normalize, similar, tokens
from tests.memory_extraction_support import candidate


@pytest.mark.parametrize(
    ("kind", "left", "right", "expected"),
    [
        ("metric_override", {"metric_key": "gmv", "patch": {}},
         {"metric_key": "gmv", "patch": {"date_field": None}}, "duplicate"),
        ("metric_override", {"metric_key": "gmv", "patch": {}},
         {"metric_key": "refund_rate", "patch": {}}, "distinct"),
        ("metric_override", {"metric_key": "gmv", "patch": {}},
         {"metric_key": "gmv", "patch": {"date_field": "paid_at"}}, "conflict"),
        ("metric_override", {"metric_key": "gmv", "patch": {"add_filters": ["a", "b"]}},
         {"metric_key": "gmv", "patch": {"add_filters": ["b", "a"]}}, "conflict"),
        ("region_focus", {"region_ids": [1, 2, 1]}, {"region_ids": [2, 1]}, "duplicate"),
        ("region_focus", {"region_ids": [1]}, {"region_ids": [2]}, "conflict"),
        ("terminology", {"term": " ＧＭＶ ", "means": "退款申请时间"},
         {"term": "gmv", "means": "退款申请的时间"}, "duplicate"),
        ("terminology", {"term": "大促", "means": "618"},
         {"term": "小促", "means": "618"}, "distinct"),
        ("terminology", {"term": "大促", "means": "618"},
         {"term": "大促", "means": "双11"}, "conflict"),
        ("format_preference", {"prefer": "table", "decimals": 2},
         {"prefer": "table", "decimals": 2}, "duplicate"),
        ("format_preference", {"prefer": "table", "decimals": 2},
         {"prefer": "prose", "decimals": 2}, "conflict"),
        ("format_preference", {"prefer": "table", "decimals": 2},
         {"prefer": "table", "decimals": 3}, "conflict"),
    ],
)
def test_type_specific_comparison(
    kind: str, left: dict[str, object], right: dict[str, object], expected: str
) -> None:
    a = candidate(memory_type=kind, content=left)
    b = candidate(memory_type=kind, content=right, confidence=0.7, summary="another source")
    before = a.model_dump_json(), b.model_dump_json()
    assert compare(a, b) is MemoryMatch(expected)
    assert (a.model_dump_json(), b.model_dump_json()) == before


def test_different_types_are_distinct() -> None:
    assert compare(candidate(), candidate(memory_type="region_focus", content={"region_ids": [1]})) is MemoryMatch.DISTINCT


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("a b c d", "a b c d e", True),
        ("a b c", "a b c d", False),
        ("退 款 申 请", "退款申请的", True),
        ("退款", "不退款", False),
        ("618", "619", False),
        ("！！", "!!", True),
        ("!", "?", False),
        ("!", "a", False),
        ("a", "?", False),
        ("  ", "\t", True),
    ],
)
def test_jaccard_boundary_and_empty_tokens(left: str, right: str, expected: bool) -> None:
    assert similar(left, right) is expected


def test_normalization_preserves_negation_and_numbers() -> None:
    assert normalize("  ＧＭＶ\t RATE  ") == "gmv rate"
    assert tokens("不退款，NOT ６１８；GMV！") == {"不", "退", "款", "not", "618", "gmv"}


@pytest.mark.parametrize("status", list(WriteStatus))
def test_write_outcome_predecessor_contract(status: WriteStatus) -> None:
    current, old = uuid4(), uuid4()
    expected = old if status is WriteStatus.SUPERSEDED else None
    assert WriteOutcome(status=status, memory_id=current, superseded_id=expected).status is status
    with pytest.raises(ValidationError):
        WriteOutcome(status=status, memory_id=current, superseded_id=None if expected else old)
    with pytest.raises(ValidationError):
        WriteOutcome(status=status, memory_id=current, superseded_id=current)


def test_memory_quota_defaults_and_examples() -> None:
    settings = RateLimitSettings()
    assert settings.memories_rules == settings.conversations_rules
    assert settings.memories_rules is not settings.conversations_rules
    for name in (".env.example", ".env.api.container.example"):
        assert "IP_RATE_LIMITS__MEMORIES_RULES=" in Path(name).read_text()
    for rules in ([], [RateRule(requests=1, seconds=60)] * 11):
        with pytest.raises(ValidationError):
            RateLimitSettings(memories_rules=rules)
