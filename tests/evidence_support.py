"""Durable knowledge answer fixtures shared by source-mutation acceptance tests."""

from dataclasses import dataclass, replace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from langgraph.runtime import Runtime

from app.agents.contracts import EvidenceBundle, Route, RouteDecision, TurnIdentity
from app.agents.nodes.format_answer import format_answer
from app.agents.state import AgentState, GraphOutput
from app.api.dependencies import get_current_user
from app.application import create_app
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.db.models import Turn
from app.db.session import Database
from app.retrieval.consistency_store import ConsistencyStore
from app.schemas.auth import UserResponse
from app.schemas.ingestion import ActiveManifest, PreparedDocument, VectorRow
from app.schemas.knowledge import KnowledgeEvidence
from app.services.consistency import ConsistencyService
from app.services.evidence import EvidenceService
from app.services.ingestion import IngestionService
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import IngestionPlan
from app.services.knowledge_generation import KnowledgeGenerationService
from tests.agents.support import context
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import admitted
from tests.knowledge_support import draft
from tests.retrieval_support import RetrievalHarness


@dataclass
class HistoricalKnowledge:
    """One committed answer and its independently retrievable PostgreSQL evidence."""

    app: FastAPI
    identity: TurnIdentity
    bundle: EvidenceBundle
    answer: dict[str, object]

    async def assert_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise actual history routes with all external service methods forbidden."""
        for name in ("mcp", "llm", "retrieval", "model_runtime", "knowledge_generation"):
            service = AsyncMock()
            for method in (
                "call_tool",
                "generate_structured",
                "retrieve",
                "embed",
                "rerank",
                "generate",
            ):
                setattr(
                    service, method, AsyncMock(side_effect=AssertionError("external history call"))
                )
            monkeypatch.setattr(self.app.state, name, service)
        # A new service instance must read the same persisted records.
        self.app.state.evidence = EvidenceService(self.app.state.database)
        base = f"/api/v1/conversations/{self.identity.conversation_id}/turns"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            response = await client.get(f"{base}/{self.identity.turn_id}/evidence")
            assert response.status_code == 200, response.text  # noqa: PLR2004
            body = response.json()
            assert body["data"] is None
            assert body["knowledge"] == self.bundle.knowledge.model_dump(mode="json")
            turns = await client.get(base)
            assert turns.status_code == 200  # noqa: PLR2004
            assert turns.json()["items"][-1]["answer"] == self.answer


async def committed_knowledge(
    database: Database, settings: Settings, evidence: KnowledgeEvidence
) -> HistoricalKnowledge:
    """Use the production evidence, formatting and answer commit boundaries."""
    identity = await admitted(database)
    service = EvidenceService(database)
    bundle = await service.commit_bundle(identity, None, evidence)
    llm = FakeChatModel([draft(evidence.chunks[0].chunk_id)])
    trace_id = identity.turn_id.hex
    async with database.session() as session, session.begin():
        turn = await session.get(Turn, identity.turn_id)
        turn.trace_id = trace_id
    ctx = replace(
        context(settings=settings),
        identity=identity,
        trace_id=trace_id,
        evidence=service,
        llm=llm,
        knowledge_generation=KnowledgeGenerationService(llm),
    )
    route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="退款政策")
    state = AgentState(
        **identity.model_dump(), question="退款政策", route=route, evidence_refs=bundle.refs
    )
    command = await format_answer(state, Runtime(context=ctx))
    assert command.update["status"] == "succeeded", command.update
    output = GraphOutput(
        answer=command.update["answer"],
        status=command.update["status"],
        route=route,
        evidence_refs=bundle.refs,
    )
    app = create_app(settings, database=database)

    async def user() -> UserResponse:
        return UserResponse(
            id=identity.user_id, email="audit@example.com", display_name="Audit", is_active=True
        )

    app.dependency_overrides[get_current_user] = user
    await app.state.chat._succeed(identity, output, 1)
    return HistoricalKnowledge(app, identity, bundle, output.answer.model_dump(mode="json"))


async def rebuild_index(harness: RetrievalHarness) -> str | None:
    """Use the production consistency repair, preserving the committed corpus identity."""
    settings = IngestionSettings()
    async with ConsistencyStore(harness.ingestion.settings) as store:
        ingestion = IngestionService(
            harness.database, store, harness.model, settings, harness.model_settings
        )

        async def encode(
            prepared: list[PreparedDocument], manifest: ActiveManifest, budget: Deadline
        ) -> list[VectorRow]:
            rows, _ = await ingestion.encode(IngestionPlan(changed=prepared), manifest, budget)
            return rows

        report = await ConsistencyService(harness.database, store, settings, encode).check(
            root=harness.root
        )
        assert report.successful, report
        assert report.chunks_inserted > 0
        return report.corpus_version
