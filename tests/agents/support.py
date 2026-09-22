"""Offline graph fixtures never substitute for durable PostgreSQL acceptance."""

import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from app.agents.contracts import (
    DataEvidence,
    EvidenceBundle,
    EvidenceSnapshot,
    KnowledgeSnapshot,
    PreparedContext,
    SqlGeneratorOutput,
    TurnIdentity,
)
from app.agents.graph import RECURSION_LIMIT, topology
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphInput, GraphOutput
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import ConflictError
from app.schemas.clarification import AvailableMetric, ClarificationCapabilities
from app.schemas.corpus import DocumentType
from app.schemas.knowledge import KnowledgeEvidence
from app.schemas.mcp import QueryResultPayload
from app.schemas.metric_resolution import MetricIntent
from app.schemas.metrics import MetricDefinition
from app.schemas.schema_catalog import SchemaCatalog
from app.services.regions import RegionService
from app.services.schema_tokens import SchemaTokenCounter
from data.seed.metrics_loader import load_catalog as load_metrics
from data.seed.schema_metadata_loader import load_catalog as load_schema
from tests.answer_support import data_draft
from tests.factories import business_schema
from tests.factories import query_result as result
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient


class FakeEvidence:
    def __init__(self) -> None:
        self.snapshot: EvidenceSnapshot | None = None
        self.committed = False
        self.knowledge: KnowledgeSnapshot | None = None

    async def find(
        self, identity: TurnIdentity, snapshot_id: UUID | None = None
    ) -> EvidenceSnapshot | None:
        return self.snapshot

    async def commit(self, identity: TurnIdentity, data: DataEvidence) -> EvidenceSnapshot:
        if self.snapshot and self.snapshot.data != data:
            raise ConflictError()
        self.snapshot = self.snapshot or EvidenceSnapshot(id=uuid4(), data=data)
        self.committed = True
        return self.snapshot

    async def read_bundle(self, identity: TurnIdentity) -> EvidenceBundle:
        return EvidenceBundle(data=self.snapshot, knowledge=self.knowledge)

    async def commit_bundle(
        self, identity: TurnIdentity, data: DataEvidence | None, knowledge: KnowledgeEvidence | None
    ) -> EvidenceBundle:
        if knowledge is not None and self.knowledge and self.knowledge.knowledge != knowledge:
            raise ConflictError()
        if data is not None:
            await self.commit(identity, data)
        if knowledge is not None:
            self.knowledge = self.knowledge or KnowledgeSnapshot(id=uuid4(), knowledge=knowledge)
            self.committed = True
        return await self.read_bundle(identity)


class FakeConversations:
    async def prepare(self, identity: TurnIdentity) -> PreparedContext:
        return PreparedContext(question="2026年8月GMV", summary="", messages=[], prior_sql=[])


class FakeSchemaCatalog:
    async def render(
        self, tables: list[str] | None = None, *, deadline: Deadline | None = None
    ) -> str:
        return "test schema"

    async def snapshot(self, *, deadline: Deadline | None = None) -> SchemaCatalog:
        return load_schema(Path("data/seed/schema_metadata.yaml"))


class FakeMetrics:
    async def list_active(self, *, deadline: Deadline | None = None) -> list[MetricDefinition]:
        return load_metrics(Path("data/seed/metrics.yaml")).definitions

    async def get_active(self, key: str, *, deadline: Deadline | None = None) -> MetricDefinition:
        return next(item for item in await self.list_active(deadline=deadline) if item.key == key)


class FakeClarificationCapabilities:
    async def read(self, *, deadline: Deadline) -> ClarificationCapabilities:
        return ClarificationCapabilities(
            metrics=[
                AvailableMetric(key=item.key, display_name=item.display_name)
                for item in await FakeMetrics().list_active(deadline=deadline)
            ],
            document_categories=list(DocumentType),
        )


def metric_intent() -> MetricIntent:
    return MetricIntent(metric_keys=["gmv"], period_expression="2026年8月", grain="total")


def sql_candidate(sql: str = "SELECT 42") -> SqlGeneratorOutput:
    return SqlGeneratorOutput(thinking="Scripted query", sql=sql, tables_used=[])


def context(
    *,
    responses: list[BaseModel | Exception] | None = None,
    mcp_results: list[QueryResultPayload | Exception] | None = None,
    settings: Settings | None = None,
) -> RuntimeContext:
    settings = settings or Settings(
        _env_file=None,
        database={"app_password": "test-password"},
        security={"jwt_secret": "test-secret-at-least-32-characters-long"},
        mcp={"auth_token": "test-mcp"},
        llm={"base_url": "https://example.invalid/v1", "model": "test", "api_key": "test-key"},
        router={"strategy": "hybrid"},  # Legacy scripts explicitly exercise this comparison arm.
    )
    mcp = FakeMcpClient(
        [result()] if mcp_results is None else mcp_results,
        schema_responses=[business_schema() for _ in range(10)],
    )
    return RuntimeContext(
        regions=RegionService(mcp),
        metrics=FakeMetrics(),
        clarification_capabilities=FakeClarificationCapabilities(),
        now=datetime(2026, 9, 8, tzinfo=UTC),
        schema_catalog=FakeSchemaCatalog(),
        schema_token_counter=SchemaTokenCounter(),
        settings=settings,
        deadline=Deadline(time.monotonic() + 30),
        identity=TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4()),
        trace_id="test-trace",
        llm=FakeChatModel(
            responses
            if responses is not None
            else [
                metric_intent(),
                sql_candidate(),
                data_draft(markdown="42 orders", confidence=0.9),
            ]
        ),
        mcp=mcp,
        evidence=FakeEvidence(),
        conversations=FakeConversations(),
    )


async def invoke(
    ctx: RuntimeContext, callbacks: list[BaseCallbackHandler] | None = None
) -> GraphOutput:
    graph = topology().compile(name="offline-behavior-test")
    return GraphOutput.model_validate(
        await graph.ainvoke(
            GraphInput(**ctx.identity.model_dump()),
            {"recursion_limit": RECURSION_LIMIT, "callbacks": callbacks or []},
            context=ctx,
        )
    )
