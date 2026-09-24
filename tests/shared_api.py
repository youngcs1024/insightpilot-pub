"""Transaction-isolated API clients; dedicated concurrency tests use real commits."""

from collections.abc import AsyncIterator
from http import HTTPStatus
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncConnection

from app.application import create_app
from app.core.config_models import Settings
from app.services.graph import GraphService
from tests import factories
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.shared_database import TestPostgres, TransactionDatabase


@pytest.fixture
async def client(
    settings: Settings,
    db_connection: AsyncConnection,
    fake_llm: FakeChatModel,
    fake_mcp: FakeMcpClient,
    migrated_db: TestPostgres,
) -> AsyncIterator[httpx.AsyncClient]:
    """Exercise real services and repositories without starting external lifespans."""
    settings.database = migrated_db.app
    database = TransactionDatabase(settings.database, db_connection)
    application = create_app(
        settings, database=database, graph_service=AsyncMock(spec=GraphService), mcp_client=fake_mcp
    )
    application.state.llm = fake_llm
    # This savepoint fixture shares one connection; background commits use the
    # dedicated real-commit memory/chat suites, not concurrent savepoints here.
    application.state.chat.memory.run = AsyncMock(return_value=None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
        timeout=10,
    ) as resource:
        yield resource


@pytest.fixture
async def auth_client(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Register and log in a real fixture user, keeping token validation enabled."""
    email = factories.user().email
    credentials = {"email": email, "password": "Fixture-pass-123!"}
    registered = await client.post(
        "/api/v1/auth/register",
        json={**credentials, "display_name": "Fixture user"},
    )
    assert registered.status_code == HTTPStatus.CREATED, registered.text
    logged_in = await client.post("/api/v1/auth/login", json=credentials)
    assert logged_in.status_code == HTTPStatus.OK, logged_in.text
    client.headers["Authorization"] = f"Bearer {logged_in.json()['access_token']}"
    return client
