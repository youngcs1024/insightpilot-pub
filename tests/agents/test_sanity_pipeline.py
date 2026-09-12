"""Advisories survive commit and rendering without another SQL or MCP call."""

# ruff: noqa: PLR2004 -- explicit result-shape and call-count acceptance examples.

import json
from collections.abc import Iterator
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph

from app.agents.contracts import (
    MAX_ANSWER_CHARS,
    AnswerDraft,
    DataEvidence,
    EvidenceSnapshot,
)
from app.agents.data import sanity
from app.agents.data.caveats import CAVEATS
from app.agents.data.nodes.sanity_check import sanity_check
from app.agents.data.state import DataAgentState
from app.agents.runtime import RuntimeContext
from app.core.config_models import SanitySettings
from app.core.masking import REDACTED, mask, safe_attributes
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.mcp import QueryResultPayload, SqlValue
from app.schemas.sanity import SanityFlag
from tests.agents.support import context, invoke, metric_intent, sql_candidate
from tests.factories import sanity_payload as payload
from tests.observability_support import tracing


@pytest.mark.parametrize(
    ("rows", "flag"),
    [
        ([], SanityFlag.EMPTY_RESULT),
        ([[None], [None]], SanityFlag.ALL_NULL),
        ([[None]], SanityFlag.SINGLE_NULL_SCALAR),
        ([[1]], SanityFlag.TRUNCATED),
        ([[0]], SanityFlag.SUSPICIOUS_ZERO),
        ([["1000000000001"]], SanityFlag.EXTREME_MAGNITUDE),
        ([[-1]], SanityFlag.NEGATIVE_MONEY),
        ([[1], [2]], SanityFlag.CARDINALITY_SPIKE),
    ],
)
async def test_every_flag_is_committed_rendered_and_never_retried(
    rows: list[list[SqlValue]], flag: SanityFlag
) -> None:
    result = payload(rows, truncated=flag is SanityFlag.TRUNCATED)
    ctx = context(
        responses=[
            metric_intent(),
            sql_candidate("SELECT gross_amount FROM biz.orders"),
            AnswerDraft(markdown="结果见证据。", confidence=0.5),
        ],
        mcp_results=[result],
    )
    ctx.settings.data_agent.sanity = SanitySettings(
        money_columns=["amount"], nonzero_columns=["amount"], expected_max_rows=1
    )
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert not output.failures
    assert flag in ctx.evidence.snapshot.data.sanity_flags
    assert CAVEATS[flag] in output.answer.markdown
    assert output.evidence_refs.data_snapshot_id == ctx.evidence.snapshot.id
    assert len(ctx.mcp.calls) == 1
    assert len(ctx.llm.calls) == (3 if rows else 2)
    if rows:
        content = json.loads(ctx.llm.calls[-1].messages[-1].content)
        assert flag.value in content["sanity_flags"]
        assert content["generation_block"] == ctx.evidence.snapshot.data.generation_block
    restored = DataEvidence.model_validate_json(ctx.evidence.snapshot.data.model_dump_json())
    assert flag in restored.sanity_flags


async def test_check_failure_still_commits_result_and_formats_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(result: QueryResultPayload, settings: SanitySettings) -> Iterator[SanityFlag]:
        yield SanityFlag.TRUNCATED
        raise RuntimeError("synthetic inspection failure")

    monkeypatch.setattr(sanity, "result_flags", broken)
    ctx = context(mcp_results=[payload([[42]], truncated=True)])
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert output.data_evidence.rows == [[42]]
    assert CAVEATS[SanityFlag.TRUNCATED] in output.answer.markdown
    assert len(ctx.mcp.calls) == 1
    assert len(ctx.llm.calls) == 3


async def test_full_length_answer_reserves_room_for_advisories() -> None:
    ctx = context(
        responses=[
            metric_intent(),
            sql_candidate("SELECT NULL"),
            AnswerDraft(markdown="x" * MAX_ANSWER_CHARS, confidence=0.5),
        ],
        mcp_results=[payload([[None]])],
    )
    output = await invoke(ctx)
    assert output.status == "succeeded"
    assert len(output.answer.markdown) == MAX_ANSWER_CHARS
    assert CAVEATS[SanityFlag.SINGLE_NULL_SCALAR] in output.answer.markdown
    assert "回答正文已截短" in output.answer.markdown


async def test_replay_uses_committed_flags_despite_configuration_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context(mcp_results=[payload([[-1]])])
    ctx.settings.data_agent.sanity = SanitySettings(money_columns=["amount"])
    first = await invoke(ctx)
    saved = ctx.evidence.snapshot.model_dump_json()
    replay = context(responses=[AnswerDraft(markdown="历史结果。", confidence=0.5)], mcp_results=[])
    replay.evidence.snapshot = EvidenceSnapshot.model_validate_json(saved)
    checker = Mock(side_effect=AssertionError("Historical results must not be checked again"))
    monkeypatch.setattr(sanity, "result_flags", checker)
    output = await invoke(replay)
    assert output.status == "succeeded"
    assert output.evidence_refs == first.evidence_refs
    assert CAVEATS[SanityFlag.NEGATIVE_MONEY] in output.answer.markdown
    assert replay.evidence.snapshot.model_dump_json() == saved
    checker.assert_not_called()
    assert not replay.mcp.calls
    assert len(replay.llm.calls) == 1


@pytest.mark.parametrize("failed", [False, True])
async def test_node_trace_exports_only_advisories_and_check_status(failed: bool) -> None:
    ctx = context()
    service, exporter = tracing(ctx.settings)
    graph = StateGraph(DataAgentState, context_schema=RuntimeContext)
    graph.add_node("sanity_check", sanity_check)
    graph.add_edge(START, "sanity_check")
    graph.add_edge("sanity_check", END)
    result = payload([["private-invalid-number" if failed else "1000000000001"]], truncated=True)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            output = await graph.compile().ainvoke(
                DataAgentState(question="private-question", query_result=result),
                {"callbacks": [GraphTraceCallback()]},
                context=ctx,
            )
        assert output["sanity_check_result"].check_failed is failed
        service.client.flush()
        node = next(span for span in exporter.get_finished_spans() if span.name == "sanity_check")
        attributes = dict(node.attributes)
        prefix = "langfuse.observation.metadata."
        assert SanityFlag.TRUNCATED.value in json.loads(attributes[prefix + "sanity_flags"])
        assert json.loads(attributes[prefix + "sanity_check_failed"]) is failed
        assert attributes[prefix + "status"] == "succeeded"
        exported = json.dumps(attributes)
        for private in (
            "private-question",
            "private-invalid-number",
            "1000000000001",
            result.executed_sql,
        ):
            assert private not in exported
    finally:
        await service.aclose()


def test_trace_allowlist_filters_invalid_advisories_at_both_export_shapes() -> None:
    metadata = {
        "sanity_flags": ["negative_money", "private-injected-value"],
        "sanity_check_failed": False,
        "sql": "private-sql",
        "rows": [["private-result"]],
    }
    projected = mask(metadata)
    assert projected["sanity_flags"] == ["negative_money"]
    assert projected["sanity_check_failed"] is False
    assert "private" not in json.dumps(projected)
    key = "langfuse.observation.metadata"
    attributes = safe_attributes({key: json.dumps(metadata)})
    assert json.loads(attributes[key]) == projected
    flattened = safe_attributes(
        {
            key + ".sanity_flags": json.dumps(metadata["sanity_flags"]),
            key + ".sanity_check_failed": False,
        }
    )
    assert json.loads(flattened[key + ".sanity_flags"]) == ["negative_money"]
    assert json.loads(flattened[key + ".sanity_check_failed"]) is False
    assert safe_attributes(flattened) == flattened
    assert mask({"sanity_flags": "private", "sanity_check_failed": "private"}) == {
        "sanity_flags": REDACTED,
        "sanity_check_failed": REDACTED,
    }
