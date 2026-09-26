"""Typed offline candidates and a selector-backed memory port."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.core.deadline import Deadline
from app.schemas.memory import MemoryType, StoredMemory
from app.schemas.memory_retrieval import MemoryReadRequest, MemorySelection
from app.services.memory.retrieve import TokenCounter, select_memories

USER = UUID("00000000-0000-0000-0000-000000000001")


def stored(
    kind: MemoryType = MemoryType.TERMINOLOGY, *, user: UUID = USER,
    content: dict[str, object] | None = None, summary: str = "Saved preference",
) -> StoredMemory:
    defaults = {
        MemoryType.TERMINOLOGY: {"term": "大促", "means": "618"},
        MemoryType.METRIC_OVERRIDE: {"metric_key": "refund_rate", "patch": {"date_field": "r.requested_at"}},
        MemoryType.REGION_FOCUS: {"region_ids": [1]},
        MemoryType.FORMAT_PREFERENCE: {"prefer": "table", "decimals": 3},
    }
    return StoredMemory.model_validate({
        "id": uuid4(), "user_id": user, "source_turn_id": uuid4(), "memory_type": kind,
        "content": content or defaults[kind], "summary": summary, "confidence": 0.9,
        "created_at": datetime(2026, 9, 1, tzinfo=UTC), "updated_at": datetime(2026, 9, 1, tzinfo=UTC),
    })


class MemoryPort:
    def __init__(self, rows: list[StoredMemory], *, fail_at: int | None = None) -> None:
        self.rows = rows
        self.calls: list[MemoryReadRequest] = []
        self.fail_at = fail_at

    async def retrieve(
        self, request: MemoryReadRequest, *, deadline: Deadline, counter: TokenCounter,
    ) -> MemorySelection:
        deadline.check("fake_memory")
        self.calls.append(request.model_copy(deep=True))
        if self.fail_at == len(self.calls):
            return MemorySelection(failed=True, failure_code="database_error")
        return select_memories(self.rows, request, counter)
