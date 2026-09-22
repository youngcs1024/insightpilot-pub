"""All stack setup is fixture-owned, never triggered during collection."""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from tests.e2e.client import Session
from tests.e2e.stack import E2EStack, e2e_stack

__all__ = ["e2e_stack"]


@pytest.fixture
async def session(e2e_stack: E2EStack, request: pytest.FixtureRequest) -> AsyncIterator[Session]:
    directory = e2e_stack.directory / request.node.name
    directory.mkdir()
    async with (httpx.AsyncClient(base_url=e2e_stack.api_url, timeout=120, trust_env=False) as client,
                httpx.AsyncClient(base_url=e2e_stack.inference_url, timeout=10, trust_env=False) as inference):
        credentials = {"email": f"e2e-{uuid4().hex}@example.com", "password": "E2e-password-123!"}
        response = await client.post("/api/v1/auth/register", json={**credentials, "display_name": "E2E"})
        assert response.status_code == 201, response.text
        response = await client.post("/api/v1/auth/login", json=credentials)
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        response = await client.post("/api/v1/conversations", json={"title": "E2E"})
        response.raise_for_status()
        active = Session(e2e_stack, client, inference, response.json()["id"], directory)
        try:
            yield active
        finally:
            result = await active.status()
            (directory / "final-script.json").write_text(result.model_dump_json(indent=2))
