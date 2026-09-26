"""Shared intent extraction, independent of graph state and preference storage."""

from datetime import datetime
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agents.prompts import METRIC_INTENT
from app.core.deadline import Deadline
from app.core.llm_config import ModelRole
from app.schemas.intent import MetricIntentInput
from app.schemas.metric_resolution import MetricIntent
from app.schemas.metrics import MetricDefinition
from app.services.metric_templates import render_catalog_block
from app.services.periods import build_date_context


class IntentLlmPort(Protocol):
    async def generate_structured[T: BaseModel](
        self, role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline,
    ) -> T: ...


class IntentCatalogPort(Protocol):
    async def list_active(self, *, deadline: Deadline | None = None) -> list[MetricDefinition]: ...


def intent_messages(
    inputs: MetricIntentInput, definitions: list[MetricDefinition], *, now: datetime,
) -> list[BaseMessage]:
    """Reuse the established prompt and catalog, without saved metric/region defaults."""
    return [
        SystemMessage(content="\n\n".join([
            METRIC_INTENT, build_date_context(now=now), render_catalog_block(definitions),
        ])),
        HumanMessage(content=inputs.model_dump_json()),
    ]


class MetricIntentService:
    """Interpret once before memory eligibility; the data specialist consumes the result."""

    def __init__(self, llm: IntentLlmPort, metrics: IntentCatalogPort) -> None:
        self.llm = llm
        self.metrics = metrics

    async def interpret(
        self, inputs: MetricIntentInput, *, deadline: Deadline, now: datetime,
    ) -> MetricIntent:
        deadline.check("metric_intent")
        definitions = await self.metrics.list_active(deadline=deadline)
        return await self.llm.generate_structured(
            ModelRole.SQL, intent_messages(inputs, definitions, now=now),
            MetricIntent, deadline=deadline,
        )
