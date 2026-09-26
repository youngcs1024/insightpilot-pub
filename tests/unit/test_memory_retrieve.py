"""Hard negative gates, deterministic combined budgets and explicit priority."""

# ruff: noqa: PLR2004 -- exact memory/token cap acceptance values.

from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from app.core.errors import ConflictError
from app.schemas.memory import MemoryType
from app.schemas.memory_retrieval import MemoryReadRequest, MemoryReason, MemoryStage
from app.schemas.metric_resolution import MetricPatch, MetricPatches, MetricPatchEntry
from app.services.memory.retrieve import effective_patch, memory_text, select_memories, term_present
from app.services.schema_tokens import SchemaTokenCounter
from tests.memory_retrieval_support import USER, stored


def request(**changes: object) -> MemoryReadRequest:
    return MemoryReadRequest.model_validate({
        "user_id": USER, "question": "2026年8月大促退款率", "stage": "finalize",
        "data_route": True, "metric_keys": ["refund_rate"], "region_mentioned": False, **changes,
    })


def test_metric_override_not_retrieved_for_knowledge_question() -> None:
    result = select_memories([stored(MemoryType.METRIC_OVERRIDE)], request(data_route=False), SchemaTokenCounter())
    assert result.selected == []
    assert result.decisions[0].reason is MemoryReason.WRONG_ROUTE


def test_metric_must_match_current_intent() -> None:
    result = select_memories([stored(MemoryType.METRIC_OVERRIDE)], request(metric_keys=["gmv"]), SchemaTokenCounter())
    assert result.decisions[0].reason is MemoryReason.WRONG_METRIC


def test_region_focus_not_retrieved_when_region_stated() -> None:
    result = select_memories([stored(MemoryType.REGION_FOCUS)], request(region_mentioned=True), SchemaTokenCounter())
    assert result.selected == []
    assert result.decisions[0].reason is MemoryReason.EXPLICIT_REGION


def test_region_focus_retrieved_when_region_absent() -> None:
    row = stored(MemoryType.REGION_FOCUS)
    assert select_memories([row], request(), SchemaTokenCounter()).selected == [row]


def test_unknown_region_is_not_absence() -> None:
    result = select_memories([stored(MemoryType.REGION_FOCUS)], request(region_mentioned=None), SchemaTokenCounter())
    assert result.decisions[0].reason is MemoryReason.UNKNOWN_REGION


@pytest.mark.parametrize(("term", "question", "expected"), [
    ("大促", "今年大促政策", True), ("大促", "退款政策", False),
    ("GMV", "gmv是多少", True), ("GMV", "xgmvx", False),
    ("sale event", "SALE   EVENT rules", True), (" ", "anything", False),
])
def test_terminology_retrieved_only_when_term_present(term: str, question: str, expected: bool) -> None:
    assert term_present(term, question) is expected
    row = stored(content={"term": term, "means": "test"})
    assert bool(select_memories([row], request(question=question), SchemaTokenCounter()).selected) is expected


@pytest.mark.parametrize("stage", list(MemoryStage))
def test_format_preference_always_eligible(stage: MemoryStage) -> None:
    row = stored(MemoryType.FORMAT_PREFERENCE)
    assert select_memories([row], request(stage=stage, clarify=True), SchemaTokenCounter()).selected == [row]


def test_route_precedes_metric_gate() -> None:
    rows = [stored(kind) for kind in MemoryType]
    result = select_memories(rows, request(stage=MemoryStage.PREPARE), SchemaTokenCounter())
    assert {r.memory_type for r in result.selected} == {MemoryType.TERMINOLOGY, MemoryType.FORMAT_PREFERENCE}
    assert sum(d.reason is MemoryReason.WRONG_STAGE for d in result.decisions) == 2


def test_cap_enforced_and_stable() -> None:
    rows = [stored(content={"term": f"term{i}", "means": "meaning"}) for i in range(8)]
    inputs = request(question=" ".join(f"term{i}" for i in range(8)))
    result = select_memories(rows, inputs, SchemaTokenCounter())
    assert len(result.selected) == 5
    assert sum(d.reason is MemoryReason.COUNT_LIMIT for d in result.decisions) == 3
    assert select_memories(list(reversed(rows)), inputs, SchemaTokenCounter()) == result


@pytest.mark.parametrize(("cost", "selected"), [(600, 1), (601, 0)])
def test_exact_token_boundary(cost: int, selected: int) -> None:
    class Counter:
        def count(self, text: str) -> int:
            return cost
    result = select_memories([stored()], request(), Counter())
    assert len(result.selected) == selected


def test_combined_memory_budget_not_doubled() -> None:
    rows = [stored(content={"term": f"词{i}", "means": "解释"}) for i in range(4)]
    rows.extend([stored(MemoryType.FORMAT_PREFERENCE), stored(MemoryType.METRIC_OVERRIDE), stored(MemoryType.REGION_FOCUS)])
    inputs = request(question="词0词1词2词3退款率")
    before = select_memories(rows, inputs.model_copy(update={"stage": MemoryStage.PREPARE}), SchemaTokenCounter())
    final = select_memories(rows, inputs, SchemaTokenCounter())
    assert len(before.selected) == 5
    assert len(final.selected) == 5
    assert final.tokens == SchemaTokenCounter().count(memory_text(final.selected)) <= 600


def test_real_token_cap_skips_whole_large_candidate() -> None:
    row = stored(MemoryType.METRIC_OVERRIDE, content={"metric_key": "refund_rate", "patch": {"expression": "界" * 1900}})
    small = stored(MemoryType.FORMAT_PREFERENCE)
    result = select_memories([row, small], request(), SchemaTokenCounter())
    assert result.selected == [small]
    assert row.content.patch.expression == "界" * 1900
    assert any(d.reason is MemoryReason.TOKEN_LIMIT for d in result.decisions)


def test_explicit_fields_override_saved_patch() -> None:
    row = stored(MemoryType.METRIC_OVERRIDE)
    inputs = request(explicit_patch=MetricPatches(items=[MetricPatchEntry(metric_key="refund_rate", patch=MetricPatch(date_field="o.paid_at"))]))
    result = select_memories([row], inputs, SchemaTokenCounter())
    assert result.selected == []
    assert result.decisions[0].reason is MemoryReason.EXPLICIT_PATCH
    assert effective_patch(MetricPatch(date_field="r.requested_at", add_filters=["r.status = 'completed'"]), MetricPatch(date_field="o.paid_at")).date_field is None


def test_superseded_memory_never_retrieved() -> None:
    row = stored().model_copy(update={"is_active": False, "superseded_by": uuid4()})
    assert select_memories([row], request(), SchemaTokenCounter()).decisions[0].reason is MemoryReason.INACTIVE


def test_cross_user_memory_never_retrieved() -> None:
    assert select_memories([stored(user=uuid4())], request(), SchemaTokenCounter()).selected == []


def test_corrupt_duplicate_logical_keys_fail_closed() -> None:
    with pytest.raises(ConflictError):
        select_memories([stored(), stored()], request(), SchemaTokenCounter())


def test_ties_use_confidence_then_recency_then_id() -> None:
    rows = [stored(content={"term": f"词{i}", "means": "same"}) for i in range(3)]
    rows[0] = rows[0].model_copy(update={"id": UUID(int=1), "confidence": 1.0})
    rows[1] = rows[1].model_copy(update={"updated_at": rows[1].updated_at + timedelta(days=1)})
    result = select_memories(rows, request(question="词0词1词2"), SchemaTokenCounter())
    assert [r.id for r in result.selected] == [r.id for r in rows]
