"""Prepare an explicit query; natural-language resolution belongs to Step 3.11."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.state import KnowledgeAgentState
from app.agents.runtime import RuntimeContext
from app.schemas.retrieval import RetrievalQuery


async def rewrite_query(
    state: KnowledgeAgentState, runtime: Runtime[RuntimeContext]
) -> Command[str]:
    """Preserve the caller's time scope and use no LLM or history-based rewrite."""
    runtime.context.deadline.check("knowledge_query")
    return Command(
        update={
            "query": RetrievalQuery(
                standalone=state.knowledge_intent
                if state.knowledge_intent.strip()
                else state.question,
                time_scope=state.time_scope.model_copy(deep=True),
            )
        }
    )
