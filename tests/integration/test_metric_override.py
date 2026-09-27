"""Step 6.5: genuine cross-session memory changes executed SQL and immutable evidence."""

import json
from collections.abc import Iterator
from decimal import Decimal
from http import HTTPStatus

import httpx
import pytest

from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings, RouterSettings, Settings
from app.core.deadline import Deadline
from app.repositories.memory import MemoryRepository
from app.schemas.chat import EvidenceResponse, TurnResponse
from app.schemas.memory import MemoryType
from app.schemas.memory_retrieval import MemoryReadRequest, MemoryStage
from app.schemas.metric_resolution import MetricPatch
from app.services.memory.service import MemoryService
from app.services.metric_override import resolve_binding
from app.services.schema_tokens import SchemaTokenCounter
from tests.database_support import DatabaseStack
from tests.integration.catalog_support import catalog_migrated, client
from tests.integration.mcp_support import running_mcp_endpoint
from tests.memory_support import memory_input
from tests.metric_override_support import (
    chat_turn,
    register,
    saved_memory,
    seed_boundary_case,
    session,
)
from tests.metric_resolution_support import override, request, schema

pytestmark = pytest.mark.integration
__all__ = ["catalog_migrated", "client"]


@pytest.fixture
def server_endpoint(database_stack: DatabaseStack) -> Iterator[MCPSettings]:
    with running_mcp_endpoint(database_stack) as endpoint:
        yield endpoint


async def evidence(http: httpx.AsyncClient, turn: TurnResponse) -> EvidenceResponse:
    response = await http.get(
        f"/api/v1/conversations/{turn.conversation_id}/turns/{turn.id}/evidence"
    )
    assert response.status_code == HTTPStatus.OK, response.text
    return EvidenceResponse.model_validate(response.json())


async def test_override_changes_date_field_in_sql(
    settings: Settings,
    catalog_migrated: None,
    database_stack: DatabaseStack,
    server_endpoint: MCPSettings,
) -> None:
    customer = await seed_boundary_case(database_stack)
    settings.database = DatabaseSettings(
        port=database_stack.settings.db_host_port,
        app_password=database_stack.settings.bootstrap.app_password,
    )
    settings.router = RouterSettings()
    async with session(settings, server_endpoint) as (http, application, database):
        user = await register(http, application)
        first = await chat_turn(http, application, customer, paid=True, explicit=True, save=True)
        saved = await saved_memory(database, user)
        assert saved.source_turn_id == first.id
    # New application, database pool, MCP client and conversation; no transient state survives.
    async with session(settings, server_endpoint, user=user) as (http, application, database):
        second = await chat_turn(http, application, customer, paid=True, explicit=False)
        snapshot = await evidence(http, second)
        assert first.conversation_id != second.conversation_id
        binding = snapshot.data.data.metric_bindings[0]
        assert binding.override_id == saved.id
        assert binding.date_field == "o.paid_at"
        assert "o.paid_at >=" in snapshot.data.data.sql
        assert "r.requested_at >=" not in snapshot.data.data.sql
        assert Decimal(snapshot.data.data.rows[0][1]) == 1
        assert snapshot.data.id == second.evidence_refs.data_snapshot_id
        assert saved.created_at.date().isoformat() in second.answer.markdown
        assert "订单支付时间" in second.answer.markdown
        assert "自定义定义" in second.answer.markdown
        explicit = await chat_turn(http, application, customer, paid=False, explicit=True)
        default_snapshot = await evidence(http, explicit)
        assert default_snapshot.data.data.metric_bindings[0].override_id is None
        assert default_snapshot.data.data.metric_bindings[0].date_field == "r.requested_at"
        assert Decimal(default_snapshot.data.data.rows[0][1]) == 0
        # Commit a later supersession and prove historical evidence/replay remain unchanged.
        async with database.session() as db, db.begin():
            repo = MemoryRepository(db, user.id)
            replacement = await repo.create(
                memory_input(
                    explicit.id,
                    kind=MemoryType.METRIC_OVERRIDE,
                    content={
                        "metric_key": "refund_rate",
                        "patch": {"date_field": "r.requested_at"},
                    },
                )
            )
            await repo.supersede(saved.id, by=replacement.id)
        assert await evidence(http, second) == snapshot
        replay = await http.post(
            f"/api/v1/conversations/{second.conversation_id}/messages",
            json={"content": f"客户{customer}在2026年8月的退款率是多少?"},
            headers={"Idempotency-Key": "metric-override-demo"},
        )
        assert replay.status_code == HTTPStatus.OK
        replayed = TurnResponse.model_validate(replay.json())
        assert replayed.replayed
        assert replayed.answer == second.answer
        # The ordinary gated reader must not inject this memory into knowledge questions.
        selection = await MemoryService(database, settings.database).retrieve(
            MemoryReadRequest(
                user_id=user.id,
                question="退款政策是什么",
                stage=MemoryStage.FINALIZE,
                data_route=False,
            ),
            deadline=Deadline(float("inf")),
            counter=SchemaTokenCounter(),
        )
        assert not selection.selected
        other = await register(http, application)
        foreign = await chat_turn(http, application, customer, paid=False, explicit=False)
        foreign_snapshot = await evidence(http, foreign)
        assert other.id != user.id
        assert foreign_snapshot.data.data.metric_bindings[0].override_id is None
        assert Decimal(foreign_snapshot.data.data.rows[0][1]) == 0
        print(
            json.dumps(
                {
                    "first_conversation": str(first.conversation_id),
                    "second_conversation": str(second.conversation_id),
                    "memory_id": str(saved.id),
                    "source_turn_id": str(saved.source_turn_id),
                    "evidence_id": str(snapshot.data.id),
                    "sql": snapshot.data.data.sql,
                    "assumptions": second.answer.assumptions,
                    "saved_result": "1",
                    "company_result": "0",
                },
                ensure_ascii=False,
            )
        )


@pytest.mark.parametrize(("confidence", "applied"), [(0.89, False), (0.9, True)])
async def test_expression_patch_requires_high_confidence(
    client: McpClient, confidence: float, applied: bool
) -> None:
    value = request()
    value.override = override(MetricPatch(expression="SUM(o.gross_amount)"))
    value.override.confidence = confidence
    result = await resolve_binding(value, schema(), client, deadline=Deadline(float("inf")))
    assert (result.binding.override_id is not None) == applied


async def test_invalid_column_patch_skipped_and_noted(client: McpClient) -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.nonexistent"))
    result = await resolve_binding(value, schema(), client, deadline=Deadline(float("inf")))
    assert result.binding.override_id is None
    assert result.binding.date_field == "o.paid_at"
    assert any("无法应用" in note for note in result.assumptions)
