"""Call the injected complete pipeline once, retaining its typed ranking outcome."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.state import KnowledgeAgentState
from app.agents.runtime import RuntimeContext
from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.core.observability import mark_degraded


async def retrieve(state: KnowledgeAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Reuse the caller's deadline and service-owned retry budget."""
    context = runtime.context
    context.deadline.check("knowledge_retrieve")
    if state.query is None:
        raise RetrievalConfigurationError(reason="knowledge_query_missing")
    if context.retrieval is None:
        raise RetrievalUnavailableError(operation="knowledge_service_missing")
    result = await context.retrieval.retrieve(
        state.query.model_copy(deep=True), deadline=context.deadline
    )
    context.deadline.check("knowledge_retrieved")
    if result.degradation is not None:
        mark_degraded("rerank")
    return Command(
        update={"retrieval_result": result},
        goto=(
            "no_evidence"
            if not result.candidates or result.meets_floor is False
            else "package_evidence"
        ),
    )
