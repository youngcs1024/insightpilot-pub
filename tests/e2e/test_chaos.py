"""Stop the real MCP while a BOTH turn is in progress; preserve knowledge evidence."""

import asyncio

import pytest

from tests.e2e.client import Session
from tests.e2e.contracts import BOTH_QUESTION, Scenario

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
        assert turn.answer.citations and not turn.answer.abstained
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
