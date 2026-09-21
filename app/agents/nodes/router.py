"""One bounded classification, with no specialist dispatch or evidence access."""

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.budget import SUMMARY_TOKENS, bounded_text
from app.agents.contracts import Route, RouteDecision, RouterInput
from app.agents.multiturn import trim_history
from app.agents.nodes.common import failed
from app.agents.nodes.prefilter import CLARIFICATION_QUESTION, prefilter
from app.agents.prompts import CLARIFY, ROUTER
from app.agents.runtime import RoutingRuntime, RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError, LlmStructuredOutputError
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, observe, record_route, update_current_observation
from app.services.llm.usage import collect_usage
from app.schemas.clarification import ClarificationCategory, ClarificationIntent, MissingDimension

logger = structlog.get_logger(__name__)
_SPECIALIST_COUNT = 2


def _clarify(decision: RouteDecision) -> RouteDecision:
    return RouteDecision(
        route=Route.CLARIFY,
        confidence=decision.confidence,
        decided_by=decision.decided_by,
        clarification_question=CLARIFICATION_QUESTION,
        clarification_intent=ClarificationIntent(
            category=ClarificationCategory.AMBIGUOUS_SCOPE,
            missing_dimensions=[MissingDimension.METRIC, MissingDimension.PERIOD],
        ),
    )


def _normalize(text: str) -> str:
    return "".join(text.split()).casefold()


def _scoped(decision: RouteDecision, question: str) -> RouteDecision:
    if decision.route is Route.BOTH:
        intents = {_normalize(decision.data_intent), _normalize(decision.knowledge_intent)}
        if _normalize(question) in intents or len(intents) != _SPECIALIST_COUNT:
            return _clarify(decision)
    values = decision.model_dump()
    if decision.route is not Route.CLARIFY:
        values["clarification_intent"] = None
    if decision.route not in {Route.DATA_ONLY, Route.BOTH}:
        values.update(data_intent="", metric_hints=[])
    if decision.route not in {Route.KNOWLEDGE_ONLY, Route.BOTH}:
        values["knowledge_intent"] = ""
    values["clarification_question"] = (
        decision.clarification_question.strip() or CLARIFICATION_QUESTION
        if decision.route is Route.CLARIFY
        else ""
    )
    return RouteDecision.model_validate(values)


async def _classify(inputs: RouterInput, ctx: RoutingRuntime) -> RouteDecision:
    decision = prefilter(inputs.question, inputs.routing_context)
    hit = decision is not None
    update_current_observation(TraceMetadata(prefilter_hit=hit, router_tokens=0 if hit else None))
    if decision is not None:
        return decision
    with collect_usage() as usage:
        try:
            decision = await ctx.llm.generate_structured(
                ModelRole.ROUTER,
                [
                    SystemMessage(content=ROUTER + "\n\n" + CLARIFY),
                    HumanMessage(content=json.dumps(inputs.model_dump(), ensure_ascii=False)),
                ],
                RouteDecision,
                deadline=ctx.deadline,
            )
            return decision.model_copy(update={"decided_by": "llm"})
        except LlmStructuredOutputError:
            ctx.deadline.check("router_structured_failure")
            logger.exception("router_structured_output_failed", exc_info=False)
            return RouteDecision(
                route=Route.CLARIFY,
                confidence=0,
                decided_by="llm",
                clarification_question=CLARIFICATION_QUESTION,
                clarification_intent=ClarificationIntent(
                    category=ClarificationCategory.AMBIGUOUS_SCOPE,
                    missing_dimensions=[MissingDimension.METRIC, MissingDimension.PERIOD],
                ),
            )
        finally:
            update_current_observation(TraceMetadata(router_tokens=usage.total))


async def route_question(inputs: RouterInput, ctx: RoutingRuntime) -> RouteDecision:
    """Use bounded history, a single logical model call and the original deadline."""
    with observe("router", TraceMetadata()):
        try:
            ctx.deadline.check("router")
            bounded = RouterInput(
                question=inputs.question,
                routing_context=inputs.routing_context.model_copy(
                    update={
                        "summary": bounded_text(inputs.routing_context.summary, SUMMARY_TOKENS),
                        "recent_messages": trim_history(inputs.routing_context.recent_messages),
                    },
                    deep=True,
                ),
            )
            original = await _classify(bounded, ctx)
            ctx.deadline.check("router_complete")
            decision = (
                _clarify(original)
                if original.confidence < ctx.settings.min_confidence
                else _scoped(original, inputs.question)
            )
            update_current_observation(
                TraceMetadata(
                    route=decision.route.value,
                    original_route=original.route.value,
                    confidence=original.confidence,
                    decided_by=decision.decided_by,
                )
            )
            logger.info(
                "route_decided",
                route=decision.route.value,
                original_route=original.route.value,
                confidence=original.confidence,
                decided_by=decision.decided_by,
            )
            return decision
        except InsightPilotError as exc:
            update_current_observation(TraceMetadata(code=exc.code))
            logger.exception("router_failed", code=exc.code, exc_info=False)
            raise


async def router(state: RouterInput, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Write only the future parent's route channel, without choosing a next node."""
    ctx = runtime.context
    decision = await route_question(
        state, RoutingRuntime(llm=ctx.llm, settings=ctx.settings.router, deadline=ctx.deadline)
    )
    record_route(decision.route.value)
    return Command(update={"route": decision})


async def parent_router(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Adapt the bounded router to parent failure and next-node ownership."""
    ctx = runtime.context
    try:
        if state.routing_context is None:
            raise ConflictError("missing prepared routing context")
        decision = await route_question(
            RouterInput(question=state.question, routing_context=state.routing_context),
            RoutingRuntime(llm=ctx.llm, settings=ctx.settings.router, deadline=ctx.deadline),
        )
        record_route(decision.route.value)
        return Command(update={"route": decision}, goto="finalize_context")
    except InsightPilotError as exc:
        return failed("route", state, exc)
