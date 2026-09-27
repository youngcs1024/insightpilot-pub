"""Two real conversations use fresh services; only model responses are scripted."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.agents.contracts import Route, RouteDecision, SqlGeneratorOutput, TurnIdentity
from app.api.dependencies import get_current_user
from app.application import create_app
from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings, Settings
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.schemas.auth import UserResponse
from app.schemas.chat import TurnResponse
from app.schemas.memory import MemoryType, StoredMemory
from app.schemas.memory_extraction import MemoryExtraction
from app.schemas.metric_resolution import MetricIntent, MetricPatch, MetricPatchEntry, MetricPatches
from app.schemas.synthesis import CellReference
from app.services.metric_binding import build_binding
from data.seed.schema import CUSTOMERS, ORDERS, REFUNDS, REGIONS
from tests.api.chat_support import ControlledGraph
from tests.database_support import DatabaseStack
from tests.fakes.chat_model import FakeChatModel
from tests.answer_support import data_draft
from tests.memory_extraction_support import candidate
from tests.metric_resolution_support import request, schema

DURABLE = "以后退款率都按订单支付时间计算"


async def seed_boundary_case(stack: DatabaseStack) -> int:
    """Operator-only fixture inserts disjoint rows; the API never gets this credential."""
    key = uuid4().int % 1_000_000_000 + 1_000_000_000
    config = DatabaseSettings(port=stack.settings.db_host_port, app_db="insightpilot_business",
                              app_user="postgres", app_password=stack.settings.postgres_superuser_password)
    engine = create_async_engine(config.app_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(insert(REGIONS).values(region_id=key, name=f"m{key}", name_en=f"m{key}", effective_from=datetime(2026, 1, 1, tzinfo=UTC)))
            await conn.execute(insert(CUSTOMERS).values(customer_id=key, region_id=key,
                registered_at=datetime(2026, 1, 1, tzinfo=UTC), channel="organic", phone="test",
                email=f"{key}@example.com", is_test_account=False))
            await conn.execute(insert(ORDERS).values(order_id=key, customer_id=key, region_id=key,
                created_at=datetime(2026, 8, 2, tzinfo=UTC), paid_at=datetime(2026, 8, 3, tzinfo=UTC),
                status="paid", gross_amount=100, discount_amount=0, shipping_fee=0))
            await conn.execute(insert(REFUNDS).values(refund_id=key, order_id=key,
                requested_at=datetime(2026, 9, 2, tzinfo=UTC), amount=100,
                reason_code="other", status="requested"))
    finally:
        await engine.dispose()
    return key


def turn_script(customer: int, *, paid: bool, explicit: bool) -> FakeChatModel:
    """Script the expected output, independently asserted against the actual SQL prompt."""
    patch = MetricPatch(add_filters=[f"o.customer_id = {customer}"],
                        date_field=("o.paid_at" if paid else "r.requested_at") if explicit else None)
    intent = MetricIntent(metric_keys=["refund_rate"], period_expression="2026年8月", grain="total",
        explicit_patch=MetricPatches(items=[MetricPatchEntry(metric_key="refund_rate", patch=patch)]))
    expected = request("refund_rate", patch=patch.model_copy(update={"date_field": "o.paid_at" if paid else "r.requested_at"}))
    sql = build_binding(expected, schema()).binding.resolved_expression
    return FakeChatModel([
        RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="2026年8月退款率"),
        intent,
        SqlGeneratorOutput(thinking="Frozen boundary case", sql=sql, tables_used=["biz.orders", "biz.customers", "biz.refunds"]),
        data_draft(markdown="退款率依据已执行查询计算。", reference=CellReference(row=0, column=1, value="1.00000000000000000000" if paid else "0.00000000000000000000")),
    ])


@asynccontextmanager
async def session(settings: Settings, endpoint: MCPSettings, *, user: UserResponse | None = None) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI, Database]]:
    database = Database(settings.database)
    database.start()
    mcp = McpClient(endpoint)
    application = create_app(settings, database=database, mcp_client=mcp)
    application.state.chat.graph = ControlledGraph()
    if user is not None:
        async def identity() -> UserResponse:
            return user
        application.dependency_overrides[get_current_user] = identity
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as http:
            yield http, application, database
    finally:
        await mcp.aclose()
        await database.aclose()


async def register(http: httpx.AsyncClient, application: FastAPI) -> UserResponse:
    response = await http.post("/api/v1/auth/register", json={"email": f"{uuid4().hex}@example.com", "password": "Strong-pass-123!", "display_name": "Memory"})
    assert response.status_code == 201, response.text
    user = UserResponse.model_validate(response.json())
    async def identity() -> UserResponse:
        return user
    application.dependency_overrides[get_current_user] = identity
    return user


async def chat_turn(http: httpx.AsyncClient, application: FastAPI, customer: int, *, paid: bool, explicit: bool, save: bool = False) -> TurnResponse:
    llm = turn_script(customer, paid=paid, explicit=explicit)
    application.state.llm = llm
    extraction = MemoryExtraction(candidates=[candidate(
        content={"metric_key": "refund_rate", "patch": {"date_field": "o.paid_at"}},
        evidence_quote=DURABLE, summary="退款率按订单支付时间计算")]) if save else MemoryExtraction()
    completed = asyncio.Event()
    writer = application.state.chat.memory
    writer.llm = FakeChatModel([extraction])
    original = writer.run
    async def tracked(identity: TurnIdentity) -> None:
        try:
            await original(identity)
        finally:
            completed.set()
    writer.run = tracked
    response = await http.post("/api/v1/conversations", json={})
    assert response.status_code == 201, response.text
    cid = response.json()["id"]
    question = f"客户{customer}在2026年8月的退款率是多少？"
    if save:
        question += DURABLE
    elif explicit:
        question += "本次按退款申请时间计算。"
    response = await http.post(f"/api/v1/conversations/{cid}/messages", json={"content": question}, headers={"Idempotency-Key": "metric-override-demo"})
    assert response.status_code == 200, response.text
    turn = TurnResponse.model_validate(response.json())
    assert turn.status == "succeeded", response.text
    await asyncio.wait_for(completed.wait(), timeout=10)
    assert [call.schema_name for call in llm.calls] == ["RouteDecision", "MetricIntent", "SqlGeneratorOutput", "DataAnswerDraft"]
    prompt = str(llm.calls[2].messages[0].content)
    assert ('"date_field": "o.paid_at"' if paid else '"date_field": "r.requested_at"') in prompt
    return turn


async def saved_memory(database: Database, user: UserResponse) -> StoredMemory:
    async with database.session() as db:
        rows = await MemoryRepository(db, user.id).list_active(MemoryType.METRIC_OVERRIDE)
    assert len(rows) == 1
    return rows[0]
