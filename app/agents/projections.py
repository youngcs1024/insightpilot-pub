"""The only parent-to-specialist input mappings; no I/O or preference selection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from app.agents.budget import SUMMARY_TOKENS, bounded_text
from app.agents.data.state import DataAgentInput
from app.agents.knowledge.state import KnowledgeAgentInput
from app.core.errors import ConflictError, ContextBudgetExceeded
from app.core.observability import TraceMetadata, observe
from app.schemas.memory import MemoryType, TerminologyContent, TerminologyProjection

if TYPE_CHECKING:
    from app.agents.runtime import SchemaTokenPort
    from app.agents.state import AgentState, TurnContext

MEMORY_COUNT = 5
MEMORY_TOKENS = 600


def _context(state: AgentState, counter: SchemaTokenPort) -> TurnContext:
    if state.route is None or state.context is None:
        raise ConflictError("Specialist projection requires finalized routing context")
    context = state.context
    # Count the combined selection, including types not destined for this specialist.
    # Identity/provenance stays in the parent; the bounded reference is content + summary.
    memories = json.dumps(
        [
            {
                "type": memory.memory_type.value,
                "content": memory.content.model_dump(mode="json"),
                "summary": memory.summary,
            }
            for memory in context.memories
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(context.memories) > MEMORY_COUNT or counter.count(memories) > MEMORY_TOKENS:
        raise ContextBudgetExceeded()
    return context


def _terminology(context: TurnContext) -> list[TerminologyProjection]:
    return [
        TerminologyProjection(term=memory.content.term, means=memory.content.means)
        for memory in context.memories
        if memory.memory_type is MemoryType.TERMINOLOGY
        and isinstance(memory.content, TerminologyContent)
    ]


def record_projection(
    inputs: DataAgentInput | KnowledgeAgentInput, counter: SchemaTokenPort
) -> None:
    """Measure the exact serialized input, exporting diagnostic scalars only."""
    specialist: Literal["data", "knowledge"] = (
        "data" if isinstance(inputs, DataAgentInput) else "knowledge"
    )
    with observe(
        "specialist_projection",
        TraceMetadata(
            projection_specialist=specialist,
            projection_tokens=counter.count(inputs.model_dump_json()),
            projection_tokenizer=counter.name,
        ),
    ):
        pass


def to_data_input(state: AgentState, *, token_counter: SchemaTokenPort) -> DataAgentInput:
    """Project routed inputs without knowledge evidence, history or raw memories."""
    context = _context(state, token_counter)
    route = state.route
    if route is None:
        raise ConflictError("missing route")
    inputs = DataAgentInput(
        question=state.question,
        data_intent=route.data_intent or state.question,
        metric_hints=list(route.metric_hints),
        relevant_memories=_terminology(context),
        prior_sql=list(context.prior_sql[-3:]),
        selected_overrides=context.selected_overrides.model_copy(deep=True),
        region_scope=context.region_scope.model_copy(deep=True) if context.region_scope else None,
        explicit_patch=context.explicit_patch.model_copy(deep=True),
        reference_period=(
            context.reference_period.model_copy(deep=True) if context.reference_period else None
        ),
    )
    record_projection(inputs, token_counter)
    return inputs


def to_knowledge_input(state: AgentState, *, token_counter: SchemaTokenPort) -> KnowledgeAgentInput:
    """Project only scoped knowledge context, with a bounded optional summary."""
    context = _context(state, token_counter)
    route = state.route
    if route is None:
        raise ConflictError("missing route")
    inputs = KnowledgeAgentInput(
        question=state.question,
        knowledge_intent=route.knowledge_intent or state.question,
        time_scope=context.time_scope.model_copy(deep=True),
        region_scope=context.region_scope.model_copy(deep=True) if context.region_scope else None,
        relevant_memories=_terminology(context),
        conversation_summary=bounded_text(context.summary, SUMMARY_TOKENS),
        knowledge_history=[turn.model_copy(deep=True) for turn in context.knowledge_history],
    )
    record_projection(inputs, token_counter)
    return inputs


def to_legacy_data_input(state: AgentState, *, token_counter: SchemaTokenPort) -> DataAgentInput:
    """Adapt the explicitly retained pre-routing production entrypoint only."""
    if state.prepared is None:
        raise ConflictError("missing prepared context")
    question = state.rewritten.standalone if state.rewritten else state.prepared.question
    inputs = DataAgentInput(
        question=question, data_intent=question, prior_sql=list(state.prepared.prior_sql[-3:])
    )
    record_projection(inputs, token_counter)
    return inputs
