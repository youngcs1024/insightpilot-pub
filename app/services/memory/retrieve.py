"""Conservative lexical eligibility, one combined budget, and no I/O."""

import json
import re
from collections.abc import Sequence
from typing import Protocol

from app.core.errors import ConflictError, InvalidMetricPatchError
from app.services.metric_patch_sql import canonical_filter
from app.schemas.memory import (
    Memory, MemoryType, MetricOverrideContent, StoredMemory, TerminologyContent,
)
from app.schemas.memory_retrieval import (
    MemoryDecision, MemoryReadRequest, MemoryReason, MemorySelection, MemoryStage,
)
from app.schemas.metric_resolution import MetricPatch
from app.services.memory.dedup import normalize, tokens

MAX_MEMORIES = 5
MAX_MEMORY_TOKENS = 600


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


def memory_text(memories: Sequence[Memory]) -> str:
    """Use the same exact serialization for selection and specialist projection."""
    return json.dumps(
        [{"type": row.memory_type.value, "content": row.content.model_dump(mode="json"),
          "summary": row.summary} for row in memories],
        ensure_ascii=False, separators=(",", ":"),
    )


def term_present(term: str, question: str) -> bool:
    """Match whole normalized phrases; ASCII words cannot match inside other words."""
    term, question = normalize(term), normalize(question)
    if not term:
        return False
    left = r"(?<![a-z0-9_])" if term[0].isascii() and term[0].isalnum() else ""
    right = r"(?![a-z0-9_])" if term[-1].isascii() and term[-1].isalnum() else ""
    return re.search(left + re.escape(term) + right, question) is not None


def effective_patch(saved: MetricPatch, explicit: MetricPatch) -> MetricPatch:
    """Remove explicitly owned scalar fields; binding validates SQL/filter precedence."""
    try:
        explicit_add = {canonical_filter(f) for f in explicit.add_filters}
        explicit_remove = {canonical_filter(f) for f in explicit.remove_filters}
        additions = [f for f in saved.add_filters if canonical_filter(f) not in explicit_remove]
        removals = [f for f in saved.remove_filters if canonical_filter(f) not in explicit_add]
    except InvalidMetricPatchError:
        additions, removals = list(saved.add_filters), list(saved.remove_filters)
    return saved.model_copy(update={
        "date_field": saved.date_field if explicit.date_field is None else None,
        "expression": saved.expression if explicit.expression is None else None,
        "add_filters": additions, "remove_filters": removals,
    }, deep=True)


def eligibility(row: StoredMemory, request: MemoryReadRequest) -> MemoryReason:
    """Hard gates are independent of lexical similarity and ranking."""
    if row.user_id != request.user_id:
        return MemoryReason.WRONG_USER
    if not row.is_active or row.superseded_by is not None or row.superseded_at is not None:
        return MemoryReason.INACTIVE
    if request.stage is MemoryStage.PREPARE and row.memory_type not in {
        MemoryType.TERMINOLOGY, MemoryType.FORMAT_PREFERENCE,
    }:
        return MemoryReason.WRONG_STAGE
    if isinstance(row.content, TerminologyContent):
        return (MemoryReason.SELECTED if term_present(row.content.term, request.question)
                else MemoryReason.TERM_ABSENT)
    if isinstance(row.content, MetricOverrideContent):
        if not request.data_route or request.clarify:
            return MemoryReason.WRONG_ROUTE
        if row.content.metric_key not in request.metric_keys:
            return MemoryReason.WRONG_METRIC
        patch = effective_patch(row.content.patch, request.explicit_patch.for_metric(row.content.metric_key))
        if patch == MetricPatch():
            return MemoryReason.EXPLICIT_PATCH
    if row.memory_type is MemoryType.REGION_FOCUS:
        if request.clarify:
            return MemoryReason.WRONG_ROUTE
        if request.region_mentioned is None:
            return MemoryReason.UNKNOWN_REGION
        if request.region_mentioned:
            return MemoryReason.EXPLICIT_REGION
    return MemoryReason.SELECTED


def _key(row: StoredMemory) -> tuple[MemoryType, str]:
    if isinstance(row.content, MetricOverrideContent):
        return row.memory_type, row.content.metric_key
    if isinstance(row.content, TerminologyContent):
        return row.memory_type, normalize(row.content.term)
    return row.memory_type, ""


def select_memories(
    rows: list[StoredMemory], request: MemoryReadRequest, counter: TokenCounter,
) -> MemorySelection:
    """Rank eligible rows and skip whole entries that exceed either shared cap."""
    active = [row for row in rows if row.user_id == request.user_id and row.is_active]
    keys = [_key(row) for row in active]
    if len(keys) != len(set(keys)):
        raise ConflictError("Multiple active memories for one logical key")
    query = tokens(request.question)
    scored = []
    for row in rows:
        words = tokens(memory_text([row]))
        score = len(query & words) / len(query | words) if query | words else 0.0
        scored.append((row, score))
    scored.sort(key=lambda pair: (-pair[1], -pair[0].confidence,
                                 -pair[0].updated_at.timestamp(), str(pair[0].id)))
    result = MemorySelection()
    for row, score in scored:
        reason = eligibility(row, request)
        if reason is MemoryReason.SELECTED:
            if len(result.selected) >= MAX_MEMORIES:
                reason = MemoryReason.COUNT_LIMIT
            elif counter.count(memory_text([*result.selected, row])) > MAX_MEMORY_TOKENS:
                reason = MemoryReason.TOKEN_LIMIT
            else:
                result.selected.append(row.model_copy(deep=True))
        result.decisions.append(MemoryDecision(memory_id=row.id, reason=reason, score=score))
    result.tokens = counter.count(memory_text(result.selected)) if result.selected else 0
    return result
