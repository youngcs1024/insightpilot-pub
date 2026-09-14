"""Detached scripted retrieval outcomes for the isolated knowledge specialist."""

from collections import deque

from app.agents.knowledge.graph import build
from app.agents.knowledge.state import KnowledgeAgentInput, KnowledgeAgentOutput
from app.agents.runtime import RuntimeContext
from app.core.deadline import Deadline
from app.retrieval.filtering import filter_ranked
from app.schemas.retrieval import RetrievalQuery, RetrievalResult
from tests.knowledge_support import retrieval
from tests.retrieval_support import query


class FakeRetrieval:
    """No calls are invented after the explicit response queue is exhausted."""

    def __init__(self, *responses: RetrievalResult | BaseException) -> None:
        self.responses = deque(responses)
        self.calls: list[RetrievalQuery] = []
        self.deadlines: list[Deadline] = []

    async def retrieve(self, query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult:
        deadline.check("fake_retrieval")
        self.calls.append(query.model_copy(deep=True))
        self.deadlines.append(deadline)
        value = self.responses.popleft()
        if isinstance(value, BaseException):
            raise value
        detached = value.model_copy(deep=True)
        detached.query = query.model_copy(deep=True)
        return detached


def ranked(score: float = 0.8) -> RetrievalResult:
    """Use the production filter to retain pre-filter best scores on empty results."""
    value = retrieval()
    value.retrieval_config.use_rerank = True
    value.candidates[0].scores.rerank = score
    filtered = filter_ranked(value.candidates, value.retrieval_config.filtering)
    value.candidates = filtered.candidates
    value.reranked = filtered.reranked
    value.meets_floor = filtered.meets_floor
    value.top_rerank_score = filtered.top_rerank_score
    return value


def inputs(**updates: object) -> KnowledgeAgentInput:
    return KnowledgeAgentInput.model_validate(
        {"question": query().standalone, "time_scope": query().time_scope, **updates}
    )


async def invoke(
    ctx: RuntimeContext, value: KnowledgeAgentInput | None = None
) -> KnowledgeAgentOutput:
    return KnowledgeAgentOutput.model_validate(
        await build().ainvoke(value or inputs(), context=ctx)
    )
