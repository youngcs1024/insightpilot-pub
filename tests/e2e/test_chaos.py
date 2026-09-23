"""Stop the real MCP while a BOTH turn is in progress; preserve knowledge evidence."""

# ruff: noqa: PLR2004 -- readiness statuses and two-turn history are acceptance values.

import asyncio
from decimal import Decimal

import pytest

from tests.e2e.client import Session
from tests.e2e.contracts import BOTH_QUESTION, DATA_QUESTION, FOLLOWUP_QUESTION, Scenario

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


async def test_mcp_killed_midturn(session: Session) -> None:
    await session.configure(Scenario.CHAOS)
    try:
        async with asyncio.timeout(100), asyncio.TaskGroup() as tasks:
            pending = tasks.create_task(session.ask(BOTH_QUESTION))
            await session.wait_for_knowledge()
            await asyncio.to_thread(session.stack.stop, "mcp")
            released = await session.inference.post("/_e2e/release")
            released.raise_for_status()
        turn = pending.result()
        await session.route(turn, "both")
        assert turn.status == "degraded"
        assert turn.answer.degraded_components == ["data"]
        assert "数据源当前不可用" in turn.answer.markdown
        assert turn.answer.citations
        assert not turn.answer.abstained
        assert turn.evidence_refs.data_snapshot_id is None
        assert turn.evidence_refs.knowledge_snapshot_id
        evidence = await session.evidence(turn)
        assert evidence.data is None
        assert evidence.knowledge.id == turn.evidence_refs.knowledge_snapshot_id
        history = await session.history()
        assert history.items[-1].answer == turn.answer
        await session.complete()
    finally:
        await asyncio.to_thread(session.stack.restore)


async def test_mcp_restart_midconversation(session: Session) -> None:
    await session.configure(Scenario.DATA)
    first = await session.ask(DATA_QUESTION)
    assert first.status == "succeeded"
    assert (await session.evidence(first)).data is not None
    await session.complete()
    api_id = await asyncio.to_thread(session.stack.api_container_id, "api-before-mcp-restart")
    try:
        await asyncio.to_thread(session.stack.stop, "mcp")
        unavailable = await session.client.get("/ready")
        assert unavailable.status_code == 503
        assert unavailable.json()["checks"]["mcp"] is False
        await asyncio.to_thread(session.stack.start_mcp)
        assert (
            await asyncio.to_thread(session.stack.api_container_id, "api-after-mcp-restart")
            == api_id
        )
        recovered = await session.client.get("/ready")
        assert recovered.status_code == 200
        assert recovered.json()["checks"]["mcp"] is True
        await session.configure(Scenario.FOLLOWUP)
        second = await session.ask(FOLLOWUP_QUESTION)
        await session.route(second, "data_only")
        assert second.status == "succeeded"
        evidence = await session.evidence(second)
        assert Decimal(str(evidence.data.data.rows[0][0])) == Decimal("300")
        assert first.evidence_refs.data_snapshot_id != second.evidence_refs.data_snapshot_id
        assert len((await session.history()).items) == 4
        await session.complete()
    finally:
        await asyncio.to_thread(session.stack.restore)
