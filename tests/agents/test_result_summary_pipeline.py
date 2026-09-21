"""Only the committed model view reaches answer generation, including failure paths."""

from tests.answer_support import data_draft
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.agents.budget import DATA_TOKENS
from app.agents.contracts import EvidenceSnapshot
from app.agents.data import summarize
from app.agents.data.caveats import CAVEATS
from app.agents.failures import FailureKind
from app.core.deadline import Deadline
from app.core.llm_config import ModelRole
from app.core.masking import mask, safe_attributes
from app.schemas.mcp import RESULT_CEILING, ColumnSpec
from app.schemas.sanity import SanityFlag
from tests.agents.support import context, invoke, result, metric_intent, sql_candidate
from app.schemas.synthesis import RowCountReference


async def test_exact_generation_summary_persisted() -> None:
    payload = result([[i] for i in range(RESULT_CEILING)])
    ctx = context(responses=[metric_intent(), sql_candidate(), data_draft("查询结果", reference=RowCountReference(value=payload.row_count))], mcp_results=[payload])
    generate = ctx.llm.generate_structured

    async def checked[T: BaseModel](
        role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline
    ) -> T:
        if role is ModelRole.SYNTHESIS:
            assert ctx.evidence.committed
            snapshot = ctx.evidence.snapshot
            content = json.loads(messages[-1].content)
            assert content["generation_block"] == snapshot.data.generation_block
            slot = json.dumps({"generation_block": content["generation_block"]}, ensure_ascii=False)
            assert len(slot.encode("utf-8")) <= DATA_TOKENS
            assert "rows" not in content
            assert "result_summary" not in content
            block = json.loads(content["generation_block"])
            assert block["returned_row_count"] == RESULT_CEILING
            assert len(block["sample_rows"]) <= summarize.SAMPLE_CAP
            assert block["statistics"][0][-1] == "12497500"
        return await generate(role, messages, schema, deadline=deadline)

    ctx.llm.generate_structured = checked
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert output.evidence_refs.data_snapshot_id == ctx.evidence.snapshot.id
    assert len(ctx.mcp.calls) == 1


async def test_unfittable_statistics_stop_before_commit_and_synthesis() -> None:
    payload = result()
    payload.columns[0].name = "oversized" * DATA_TOKENS
    ctx = context(responses=[metric_intent(), sql_candidate(), data_draft("查询结果", reference=RowCountReference(value=payload.row_count))], mcp_results=[payload])
    output = await invoke(ctx)
    assert output.status == "failed"
    assert output.failures[-1].kind is FailureKind.CONTEXT_BUDGET_EXCEEDED
    assert output.answer is None
    assert not ctx.evidence.committed
    assert all(call.role is not ModelRole.SYNTHESIS for call in ctx.llm.calls)
    assert len(ctx.mcp.calls) == 1


async def test_capped_result_not_presented_as_population_total() -> None:
    payload = result([[1]] * RESULT_CEILING)
    payload.result_truncated = True
    ctx = context(responses=[metric_intent(), sql_candidate(), data_draft("查询结果", reference=RowCountReference(value=payload.row_count))], mcp_results=[payload])
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert CAVEATS[SanityFlag.TRUNCATED] in output.answer.markdown
    block = json.loads(ctx.evidence.snapshot.data.generation_block)
    assert block["result_truncated"]
    assert block["statistics_scope"] == "returned_rows"
    assert block["statistics"][0][-1] == str(RESULT_CEILING)
    assert "not the uncapped population" in block["qualification"]
    assert "Preserve user top-N scope" in block["qualification"]


async def test_reused_snapshot_is_not_rebudgeted(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = result([["private-business-value" * DATA_TOKENS]])
    payload.columns = [ColumnSpec(name="description", type="text")]
    ctx = context(responses=[metric_intent(), sql_candidate(), data_draft("查询结果", reference=RowCountReference(value=payload.row_count))], mcp_results=[payload])
    first = await invoke(ctx)
    saved = ctx.evidence.snapshot.model_dump_json()
    replay = context(
        responses=[data_draft(markdown="Historical answer", confidence=1, reference=RowCountReference(value=1))], mcp_results=[]
    )
    replay.evidence.snapshot = EvidenceSnapshot.model_validate_json(saved)
    forbidden = Mock(side_effect=AssertionError("Committed evidence must not be rendered again"))
    monkeypatch.setattr(summarize, "summarize_result", forbidden)
    monkeypatch.setattr(summarize, "render_block", forbidden)
    output = await invoke(replay)
    assert output.status == "succeeded"
    assert output.evidence_refs == first.evidence_refs
    assert replay.evidence.snapshot.model_dump_json() == saved
    assert json.loads(replay.llm.calls[-1].messages[-1].content)["generation_block"] == (
        ctx.evidence.snapshot.data.generation_block
    )
    forbidden.assert_not_called()
    assert not replay.mcp.calls
    # Audit-only values, statistics and serialized prompt bodies remain private.
    data = replay.evidence.snapshot.data
    assert "private-business-value" not in json.dumps(mask(data))
    exported = safe_attributes({"langfuse.observation.output": data.model_dump_json()})
    assert "private-business-value" not in json.dumps(exported)
