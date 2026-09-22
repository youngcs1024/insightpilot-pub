"""Typed real HTTP session and request-specific observation assertions."""

import asyncio
from dataclasses import dataclass
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
        status = await self.status()
        (self.directory / (uuid4().hex+"-inference.json")).write_text(status.model_dump_json(indent=2))
        assert not status.errors, status
        assert status.remaining and not any(status.remaining.values()), status
        return status

    async def ask(self, question: str) -> TurnResponse:
        response = await self.client.post(f"/api/v1/conversations/{self.cid}/messages",
                                          json={"content": question},
                                          headers={"X-Request-ID": uuid4().hex})
        assert response.status_code == 200, response.text
        turn = TurnResponse.model_validate_json(response.content)
        (self.directory / f"{turn.id}.json").write_text(turn.model_dump_json(indent=2))
        return turn

    async def evidence(self, turn: TurnResponse) -> EvidenceResponse:
        response = await self.client.get(f"/api/v1/conversations/{self.cid}/turns/{turn.id}/evidence")
        response.raise_for_status()
        result = EvidenceResponse.model_validate_json(response.content)
        (self.directory / f"{turn.id}-evidence.json").write_text(result.model_dump_json(indent=2))
        return result

    async def history(self) -> TurnPage:
        response = await self.client.get(f"/api/v1/conversations/{self.cid}/turns")
        response.raise_for_status()
        return TurnPage.model_validate_json(response.content)

    async def observations(self) -> Observations:
        response = await self.client.get("/_e2e/observations", headers={
            "Authorization": "Bearer " + self.stack.token.get_secret_value()})
        response.raise_for_status()
        result = Observations.model_validate_json(response.content)
        (self.directory / "observations.json").write_text(result.model_dump_json(indent=2))
        return result

    async def route(self, turn: TurnResponse, expected: str) -> list[str]:
        records = [record for record in (await self.observations()).spans
                   if record.request_id == turn.trace_id]
        assert any(record.metadata.route == expected for record in records), records
        return [record.name for record in records]

    async def wait_for_knowledge(self) -> None:
        async with asyncio.timeout(40):
            while True:
                status = await self.status()
                records = (await self.observations()).spans
                if status.sql_waiting and any(record.name == "knowledge_agent"
                                             and record.metadata.status == "succeeded"
                                             for record in records):
                    return
                await asyncio.sleep(0.1)
