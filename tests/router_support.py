"""Typed offline routing fixtures shared without importing test modules."""

from time import monotonic

from pydantic import BaseModel

from app.agents.contracts import Route, RouteDecision
from app.agents.runtime import RoutingRuntime
from app.core.config_models import RouterSettings
from app.core.deadline import Deadline
from tests.fakes.chat_model import FakeChatModel

BOTH_QUESTION = "为什么华东2026年8月退款率比7月高?"


def decision(route: Route = Route.BOTH, confidence: float = 0.9) -> RouteDecision:
    return RouteDecision(
        route=route, confidence=confidence,
        data_intent="计算并比较华东2026年7月和8月退款率"
        if route in {Route.DATA_ONLY, Route.BOTH} else "",
        knowledge_intent="核查华东2026年7月至8月适用的退款政策与活动规则"
        if route in {Route.KNOWLEDGE_ONLY, Route.BOTH} else "",
        metric_hints=["refund_rate"] if route in {Route.DATA_ONLY, Route.BOTH} else [],
    )


def runtime(responses: list[BaseModel | str | Exception] | None = None) -> RoutingRuntime:
    return RoutingRuntime(
        llm=FakeChatModel(responses or []), settings=RouterSettings(),
        deadline=Deadline(monotonic() + 30),
    )
