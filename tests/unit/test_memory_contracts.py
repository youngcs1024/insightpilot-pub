"""Reserved memory slots validate content and provenance without a memory service."""

import pytest
from pydantic import ValidationError

from app.agents.state import TurnContext
from app.schemas.memory import Memory, MemoryType
from tests.agents.state_support import finalized, memory


@pytest.mark.parametrize(
    ("kind", "content"),
    [
        (MemoryType.METRIC_OVERRIDE, {"metric_key": "gmv", "patch": {"date_field": "paid_at"}}),
        (MemoryType.REGION_FOCUS, {"region_ids": [1, 2]}),
        (MemoryType.TERMINOLOGY, {"term": "大促", "means": "618"}),
        (MemoryType.FORMAT_PREFERENCE, {"prefer": "table", "decimals": 2}),
    ],
)
def test_memory_content_types_roundtrip(kind: MemoryType, content: dict[str, object]) -> None:
    value = Memory.model_validate(
        {**memory().model_dump(), "memory_type": kind, "content": content}
    )
    assert Memory.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "update",
    [
        {"memory_type": "arbitrary"},
        {"memory_type": "region_focus"},
        {"source_turn_id": None},
        {"user_id": None},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"content": {"term": "a", "means": "b" * 201}},
        {"memory_type": "region_focus", "content": {"region_ids": []}},
        {"memory_type": "region_focus", "content": {"region_ids": [True]}},
        {"memory_type": "format_preference", "content": {"prefer": "table", "decimals": 5}},
        {"content": {"term": "a", "means": "b", "instructions": "do things"}},
    ],
)
def test_memory_rejects_invalid_content_or_missing_provenance(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Memory.model_validate({**memory().model_dump(), **update})


@pytest.mark.parametrize(
    "update",
    [
        {"time_scope": None},
        {"time_scope": {"kind": "unknown"}},
        {"prior_sql": ["SELECT 1"] * 4},
        {"memories": [memory()] * 6},
        {"token_accounting": {"history": -1}},
    ],
)
def test_finalized_context_rejects_missing_scope_or_exceeded_bounds(
    update: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TurnContext.model_validate({**finalized().model_dump(), **update})
