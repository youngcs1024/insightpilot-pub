"""Explicit four-route scripts exercising the production parent and specialists."""

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from app.agents.contracts import AnswerDraft, PreparedContext, Route, RouteDecision
from app.agents.runtime import RuntimeContext
from app.services.knowledge_generation import KnowledgeGenerationService
from tests.agents.knowledge_support import FakeRetrieval, ranked
from tests.agents.support import context, metric_intent, sql_candidate
from tests.knowledge_support import draft
from tests.agents.synthesis_support import synthesis_draft

if TYPE_CHECKING:
    from app.core.config_models import Settings


def parent_context(
    route: Route,
    *,
    data_error: Exception | None = None,
    knowledge_error: Exception | None = None,
    empty: bool = False,
    settings: "Settings | None" = None,
) -> RuntimeContext:
    """Every scripted model result corresponds to a real production call."""
    retrieval = ranked(0.1 if empty else 0.8)
    decision = RouteDecision(
        route=route,
        confidence=1,
        data_intent="计算2026年8月GMV" if route in {Route.DATA_ONLY, Route.BOTH} else "",
        knowledge_intent="核查2026年8月退款政策"
        if route in {Route.KNOWLEDGE_ONLY, Route.BOTH}
        else "",
        clarification_question="请说明要查询的指标或政策。" if route is Route.CLARIFY else "",
    )
    responses = [decision]
    if route in {Route.DATA_ONLY, Route.BOTH}:
        responses.extend([metric_intent(), sql_candidate()])
        if data_error is None and route is Route.DATA_ONLY:
            responses.append(AnswerDraft(markdown="GMV 为 42。", confidence=0.9))
    if route is Route.KNOWLEDGE_ONLY and not knowledge_error and not empty:
        responses.append(draft(retrieval.candidates[0].chunk_uuid))
    if route is Route.BOTH and (data_error is None or (not knowledge_error and not empty)):
        responses.append(synthesis_draft(
            data=data_error is None,
            knowledge_id=retrieval.candidates[0].chunk_uuid
            if not knowledge_error and not empty else None,
        ))
    ctx = context(
        responses=responses, mcp_results=[data_error] if data_error else None, settings=settings
    )
    return replace(
        ctx,
        conversations=AsyncMock(
            prepare=AsyncMock(
                return_value=PreparedContext(
                    question="请分析2026年8月的经营情况", summary="", messages=[], prior_sql=[]
                )
            )
        ),
        retrieval=FakeRetrieval(knowledge_error or retrieval),
        knowledge_generation=KnowledgeGenerationService(ctx.llm),
    )
