"""Seven acceptance cases through live API/MCP and isolated real PostgreSQL/Milvus."""

import asyncio
from decimal import Decimal

import pytest

from app.schemas.synthesis import ClaimKind
from tests.e2e.client import Session
from tests.e2e.contracts import (
    BOTH_QUESTION, CLARIFY_QUESTION, DATA_QUESTION, FOLLOWUP_QUESTION, KNOWLEDGE_QUESTION, Scenario,
)

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


async def test_data_only(session: Session) -> None:
    await session.configure(Scenario.DATA)
    turn = await session.ask(DATA_QUESTION)
    calls = await session.route(turn, "data_only")
    assert turn.status == "succeeded"
    assert turn.answer.sql and turn.answer.assumptions
    assert "600" in turn.answer.markdown
    evidence = await session.evidence(turn)
    assert Decimal(str(evidence.data.data.rows[0][0])) == Decimal("600")
    assert evidence.knowledge is None
    assert "mcp_execute" in calls and "retrieve" not in calls and "retrieval_encode" not in calls
    status = await session.complete()
    assert status.embed_query == status.rerank == 0


async def test_knowledge_only(session: Session) -> None:
    await session.configure(Scenario.KNOWLEDGE)
    turn = await session.ask(KNOWLEDGE_QUESTION)
    calls = await session.route(turn, "knowledge_only")
    assert turn.status == "succeeded" and turn.answer.citations
    assert "mcp_execute" not in calls and "mcp_schema" not in calls
    assert "retrieval_encode" in calls
    evidence = await session.evidence(turn)
    assert evidence.data is None and evidence.knowledge.knowledge.chunks
    assert {citation.chunk_id for citation in turn.answer.citations} <= {
        chunk.chunk_id for chunk in evidence.knowledge.knowledge.chunks}
    status = await session.complete()
    assert status.embed_query > 0 and status.rerank > 0


async def test_both(session: Session) -> None:
    await session.configure(Scenario.BOTH)
    turn = await session.ask(BOTH_QUESTION)
    calls = await session.route(turn, "both")
    assert turn.status == "succeeded"
    assert "mcp_execute" in calls and "retrieval_encode" in calls
    evidence = await session.evidence(turn)
    assert evidence.data and evidence.knowledge
    assert [Decimal(str(row[1])) for row in evidence.data.data.rows] == [Decimal("0.5"), Decimal("1")]
    assert turn.answer.evidence_refs == turn.evidence_refs
    kinds = {claim.kind for claim in turn.answer.claims}
    assert {ClaimKind.FACT_DATA, ClaimKind.FACT_DOCUMENT, ClaimKind.INFERENCE} <= kinds
    causal = [claim for claim in turn.answer.claims if "导致" in claim.text]
    assert causal and all(claim.kind is ClaimKind.INFERENCE for claim in causal)
    assert "尚未证实因果关系" in turn.answer.markdown
    await session.complete()


async def test_clarify(session: Session) -> None:
    await session.configure(Scenario.CLARIFY)
    turn = await session.ask(CLARIFY_QUESTION)
    calls = await session.route(turn, "clarify")
    assert turn.status == "abstained" and turn.answer.abstained and turn.clarification
    assert turn.content and not turn.answer.claims
    assert turn.evidence_refs.data_snapshot_id is turn.evidence_refs.knowledge_snapshot_id is None
    assert not {"mcp_execute", "mcp_schema", "retrieval_encode"}.intersection(calls)
    await session.complete()


async def followup(session: Session, *, restart: bool) -> None:
    await session.configure(Scenario.DATA)
    first = await session.ask(DATA_QUESTION)
    initial = await session.evidence(first)
    await session.complete()
    history = await session.history()
    if restart:
        await asyncio.to_thread(session.stack.restart_api)
        assert (await session.history()).items == history.items
        assert (await session.evidence(first)).data == initial.data
    await session.configure(Scenario.FOLLOWUP)
    second = await session.ask(FOLLOWUP_QUESTION)
    await session.route(second, "data_only")
    assert second.status == "succeeded"
    evidence = await session.evidence(second)
    assert Decimal(str(evidence.data.data.rows[0][0])) == Decimal("300")
    binding = evidence.data.data.metric_bindings[0]
    assert binding.metric_key == "gmv" and binding.region_scope.region_ids == [2]
    assert binding.period_start.isoformat().startswith("2026-08-01")
    assert binding.period_end.isoformat().startswith("2026-09-01")
    assert first.evidence_refs != second.evidence_refs
    assert len((await session.history()).items) == 4
    await session.complete()


async def test_multiturn_followup(session: Session) -> None:
    await followup(session, restart=False)


async def test_restart_resumes(session: Session) -> None:
    await followup(session, restart=True)


async def test_evidence_retrievable_later(session: Session) -> None:
    await session.configure(Scenario.BOTH)
    turn = await session.ask(BOTH_QUESTION)
    original = await session.evidence(turn)
    await session.complete()
    before = await session.observations()
    try:
        await asyncio.to_thread(session.stack.stop, "mcp")
        await asyncio.to_thread(session.stack.stop, "inference")
        restored = await session.evidence(turn)
        assert restored.data == original.data and restored.knowledge == original.knowledge
        assert restored.data.data.sql and all(chunk.chunk_id for chunk in restored.knowledge.knowledge.chunks)
        assert (await session.observations()).spans == before.spans
    finally:
        await asyncio.to_thread(session.stack.restore)
