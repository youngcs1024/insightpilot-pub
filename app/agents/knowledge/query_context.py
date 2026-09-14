"""Bounded antecedent projections and deterministic scope selection, without I/O."""

import json

from app.agents.budget import HISTORY_TOKENS
from app.agents.runtime import SchemaTokenPort
from app.schemas.knowledge_query import KnowledgeHistoryTurn, KnowledgeRewrite
from app.schemas.retrieval import KnowledgeTimeScope
from app.services.knowledge_calendar import same_scope


def bounded_history(
    history: list[KnowledgeHistoryTurn], counter: SchemaTokenPort
) -> list[KnowledgeHistoryTurn]:
    """Keep a recent complete suffix; truncated-away topics cannot become antecedents."""
    selected: list[KnowledgeHistoryTurn] = []
    for turn in reversed(history):
        candidate = [turn, *selected]
        text = json.dumps([item.model_dump(mode="json") for item in candidate], ensure_ascii=False)
        if counter.count(text) > HISTORY_TOKENS:
            break
        selected = candidate
    return [turn.model_copy(deep=True) for turn in selected]


def antecedents(
    rewritten: KnowledgeRewrite, history: list[KnowledgeHistoryTurn]
) -> list[KnowledgeHistoryTurn]:
    """Unknown, missing or unresolved references are never a usable interpretation."""
    ids = set(rewritten.referenced_turn_ids)
    if rewritten.unresolved_references or not ids or not ids <= {turn.turn_id for turn in history}:
        return []
    return [turn for turn in history if turn.turn_id in ids]


def shared_scope(history: list[KnowledgeHistoryTurn]) -> KnowledgeTimeScope | None:
    """Conflicting candidate periods require clarification rather than last-item wins."""
    if not history:
        return None
    scope = history[0].time_scope
    if any(not same_scope(scope, item.time_scope) for item in history):
        return None
    return scope.model_copy(deep=True)
