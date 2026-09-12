"""Shared clients exercise real authentication and savepoint-backed services."""

from http import HTTPStatus

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User

pytestmark = pytest.mark.integration


async def test_client_starts_unauthenticated(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_auth_client_has_real_token_and_persisted_user(
    auth_client: httpx.AsyncClient,
) -> None:
    assert auth_client.headers["Authorization"].startswith("Bearer ")
    profile = await auth_client.get("/api/v1/auth/me")
    assert profile.status_code == HTTPStatus.OK
    response = await auth_client.post("/api/v1/conversations", json={})
    assert response.status_code == HTTPStatus.CREATED
    assert (await auth_client.get("/api/v1/conversations")).status_code == HTTPStatus.OK


async def test_previous_client_writes_were_rolled_back(db_session: AsyncSession) -> None:
    assert await db_session.scalar(select(func.count()).select_from(User)) == 0


async def test_client_preserves_database_error_translation(client: httpx.AsyncClient) -> None:
    body = {
        "email": "duplicate@example.com",
        "password": "Fixture-pass-123!",
        "display_name": "Test",
    }
    assert (await client.post("/api/v1/auth/register", json=body)).status_code == HTTPStatus.CREATED
    assert (
        await client.post("/api/v1/auth/register", json=body)
    ).status_code == HTTPStatus.CONFLICT
    body["email"] = "after-conflict@example.com"
    assert (await client.post("/api/v1/auth/register", json=body)).status_code == HTTPStatus.CREATED
