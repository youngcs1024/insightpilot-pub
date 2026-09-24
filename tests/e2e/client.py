"""Typed real HTTP session and request-specific observation assertions."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from uuid import uuid4

import httpx

from app.schemas.chat import EvidenceResponse, TurnPage, TurnResponse
from tests.e2e.contracts import Observations, Scenario, ScriptStatus
from tests.e2e.stack import E2EStack


@dataclass
class Session:
    stack: E2EStack
    client: httpx.AsyncClient
    inference: httpx.AsyncClient
    cid: str
    directory: Path

    async def configure(self, scenario: Scenario) -> None:
        response = await self.inference.post("/_e2e/script", json={"scenario": scenario.value})
        response.raise_for_status()

    async def status(self) -> ScriptStatus:
        response = await self.inference.get("/_e2e/status")
        response.raise_for_status()
        return ScriptStatus.model_validate_json(response.content)

    async def complete(self) -> ScriptStatus:
        # Wait for the declared post-answer call and its observed task completion
        # before changing scripts, restarting services or freezing span snapshots.
        async with asyncio.timeout(10):
            while True:
                status = await self.status()
                spans = (await self.observations()).spans
                pending = any(
                    span.name == "memory_extract" and span.metadata.status is None
                    for span in spans
                )
                if status.errors or (not any(status.remaining.values()) and not pending):
                    break
                await asyncio.sleep(0.05)
        (self.directory / (uuid4().hex + "-inference.json")).write_text(
            status.model_dump_json(indent=2)
        )
        assert not status.errors, status
        assert status.remaining, status
        assert not any(status.remaining.values()), status
        return status

    async def ask(self, question: str) -> TurnResponse:
        response = await self.client.post(
            f"/api/v1/conversations/{self.cid}/messages",
            json={"content": question},
            headers={"X-Request-ID": uuid4().hex},
        )
        assert response.status_code == HTTPStatus.OK, response.text
        turn = TurnResponse.model_validate_json(response.content)
        (self.directory / f"{turn.id}.json").write_text(turn.model_dump_json(indent=2))
        return turn

    async def evidence(self, turn: TurnResponse) -> EvidenceResponse:
        response = await self.client.get(
            f"/api/v1/conversations/{self.cid}/turns/{turn.id}/evidence"
        )
        response.raise_for_status()
        result = EvidenceResponse.model_validate_json(response.content)
        (self.directory / f"{turn.id}-evidence.json").write_text(result.model_dump_json(indent=2))
        return result

    async def history(self) -> TurnPage:
        response = await self.client.get(f"/api/v1/conversations/{self.cid}/turns")
        response.raise_for_status()
        return TurnPage.model_validate_json(response.content)

    async def observations(self) -> Observations:
        response = await self.client.get(
            "/_e2e/observations",
            headers={"Authorization": "Bearer " + self.stack.token.get_secret_value()},
        )
        response.raise_for_status()
        result = Observations.model_validate_json(response.content)
        (self.directory / "observations.json").write_text(result.model_dump_json(indent=2))
        return result

    async def route(self, turn: TurnResponse, expected: str) -> list[str]:
        records = [
            record
            for record in (await self.observations()).spans
            if record.request_id == turn.trace_id
        ]
        assert any(record.metadata.route == expected for record in records), records
        return [record.name for record in records]

    async def wait_for_knowledge(self) -> None:
        async with asyncio.timeout(40):
            while True:
                status = await self.status()
                records = (await self.observations()).spans
                if status.sql_waiting and any(
                    record.name == "knowledge_agent" and record.metadata.status == "succeeded"
                    for record in records
                ):
                    return
                await asyncio.sleep(0.1)


@asynccontextmanager
async def open_session(
    stack: E2EStack, directory: Path, *, title: str = "E2E"
) -> AsyncIterator[Session]:
    """Create a fresh authenticated user and conversation in one isolated stack."""
    directory.mkdir()
    async with (
        httpx.AsyncClient(base_url=stack.api_url, timeout=120, trust_env=False) as client,
        httpx.AsyncClient(base_url=stack.inference_url, timeout=10, trust_env=False) as inference,
    ):
        credentials = {"email": f"e2e-{uuid4().hex}@example.com", "password": "E2e-password-123!"}
        response = await client.post(
            "/api/v1/auth/register", json={**credentials, "display_name": "E2E"}
        )
        assert response.status_code == HTTPStatus.CREATED, response.text
        response = await client.post("/api/v1/auth/login", json=credentials)
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        response = await client.post("/api/v1/conversations", json={"title": title})
        response.raise_for_status()
        active = Session(stack, client, inference, response.json()["id"], directory)
        try:
            yield active
        finally:
            result = await active.status()
            (directory / "final-script.json").write_text(result.model_dump_json(indent=2))
