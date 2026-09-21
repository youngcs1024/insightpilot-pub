"""Step 4.7: immutable provenance, unified terminal rendering and bounded generation."""

# ruff: noqa: PLR2004 -- explicit confidence, precision and schema contract examples.

from dataclasses import replace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.answer_rendering import confidence_for, finalize_answer
from app.agents.answer_validation import validate_formatted_answer
from app.agents.contracts import Answer, Route, RouteDecision, RoutingContext
from app.agents.failures import FailureKind
from app.agents.nodes.finalize_context import finalize_context
from app.agents.nodes.format_answer import format_answer
from app.agents.presentation import resolve_preference
from app.agents.state import TurnContext
from app.agents.synthesis_answer import synthesis_answer
from app.core.errors import ConflictError, FabricatedCitation, LlmStructuredOutputError
from app.schemas.knowledge import Citation, KnowledgeGeneration, KnowledgePassage
from app.schemas.memory import FormatPreferenceContent
from app.schemas.sanity import SanityFlag
from app.schemas.synthesis import CellReference, ClaimKind, RowCountReference
from tests.agents.parent_support import parent_context
from tests.agents.projection_support import routed_state
from tests.agents.support import context, invoke
from tests.agents.synthesis_support import synthesis_context, synthesized
from tests.answer_support import data_draft
from tests.fakes.chat_model import FakeChatModel


async def test_quantitative_answer_includes_sql_and_assumptions() -> None:
    ctx = parent_context(Route.DATA_ONLY)
    output = await invoke(ctx)
    answer = output.answer
    data = ctx.evidence.snapshot.data
    assert answer.schema_version == 3
    assert answer.trace_id == ctx.trace_id
    assert answer.claims[0].kind is ClaimKind.FACT_DATA
    assert answer.confidence == 1
    assert "42" in answer.markdown
    assert "**统计口径**" in answer.markdown
    assert answer.markdown.endswith(f"```sql\n{data.sql}\n```")
    assert answer.assumptions == data.assumptions
    assert all(item in answer.markdown for item in data.assumptions)
    assert any("Asia/Shanghai" in item for item in answer.assumptions)
    assert len(ctx.mcp.calls) == 1 and not ctx.retrieval.calls


async def test_citations_derived_from_evidence_not_prose() -> None:
    ctx, state, bundle = await synthesis_context()
    synthesis = await synthesized(ctx, state)
    before = bundle.model_dump_json()
    answer = synthesis_answer(synthesis, bundle, [], trace_id=ctx.trace_id)
    chunk = bundle.knowledge.knowledge.chunks[0]
    assert answer.citations == [Citation.model_validate(
        chunk.model_dump(include=set(Citation.model_fields))
    )]
    assert chunk.document_title in answer.markdown
    assert bundle.model_dump_json() == before
    assert answer.claims == synthesis.claims


async def test_unresolvable_citation_is_error() -> None:
    ctx, state, bundle = await synthesis_context()
    synthesis = await synthesized(ctx, state)
    synthesis.claims[-1].chunk_ids = [uuid4()]
    with pytest.raises(FabricatedCitation):
        synthesis_answer(synthesis, bundle, [], trace_id=ctx.trace_id)


@pytest.mark.parametrize("missing", ["data", "knowledge"])
async def test_degraded_answer_names_missing_component(missing: str) -> None:
    ctx, state, bundle = await synthesis_context(data=missing != "data", knowledge=missing != "knowledge")
    answer = synthesis_answer(await synthesized(ctx, state), bundle, [], trace_id=ctx.trace_id)
    assert missing in answer.degraded_components
    assert "未能核对实际订单数据" in answer.markdown if missing == "data" else "未能核实适用政策" in answer.markdown
    assert answer.confidence <= 0.5


async def test_abstention_states_what_is_missing() -> None:
    ctx = parent_context(Route.KNOWLEDGE_ONLY, empty=True)
    answer = (await invoke(ctx)).answer
    assert answer.abstained and answer.confidence == 0
    assert answer.claims == answer.citations == []
    assert "已尝试核查企业知识库" in answer.markdown
    assert "请补充" in answer.markdown
    assert not ctx.mcp.calls


async def test_confidence_capped_when_inference_present() -> None:
    ctx = parent_context(Route.DATA_ONLY)
    answer = (await invoke(ctx)).answer
    answer.claims[0].kind = ClaimKind.INFERENCE
    assert confidence_for(answer, await ctx.evidence.read_bundle(ctx.identity)) == 0.9


@pytest.mark.parametrize("count", [0, 1, 3, 5, 8])
async def test_sanity_penalty_is_program_owned_and_clamped(count: int) -> None:
    ctx = parent_context(Route.DATA_ONLY)
    answer = (await invoke(ctx)).answer
    bundle = await ctx.evidence.read_bundle(ctx.identity)
    bundle.data.data.sanity_flags = list(SanityFlag)[:count]
    answer.claims[0].confidence = 0.01
    assert confidence_for(answer, bundle) == pytest.approx(max(0, 1 - 0.2 * count))


@pytest.mark.parametrize("route", list(Route))
async def test_all_routes_have_v3_trace_and_durable_shape(route: Route) -> None:
    ctx = parent_context(route)
    output = await invoke(ctx)
    assert output.answer.schema_version == 3
    assert output.answer.trace_id == ctx.trace_id
    if route is Route.CLARIFY:
        assert output.answer.abstained
        assert output.clarification.message in output.answer.markdown
        assert not output.answer.claims and not output.answer.sql
        assert ctx.mcp.calls == ctx.retrieval.calls == []
        assert not ctx.evidence.committed
    else:
        assert output.answer.claims
        assert output.answer.evidence_refs == output.evidence_refs
    validate_formatted_answer(
        output.answer, await ctx.evidence.read_bundle(ctx.identity), output.clarification
    )


@pytest.mark.parametrize(
    ("question", "prefer", "decimals"),
    [("请用文字回答", "prose", 3), ("保留两位小数", "table", 2),
     ("保留二位小数", "table", 2), ("请用表格，保留0位小数", "table", 0),
     ("use prose with 4 decimal places", "prose", 4),
     ("不要表格，保留一位小数", "prose", 1),
     ("use table, then use prose", "prose", 3)],
)
def test_current_explicit_format_overrides_only_its_fields(
    question: str, prefer: str, decimals: int
) -> None:
    saved = FormatPreferenceContent(prefer="table", decimals=3)
    assert resolve_preference(question, saved).model_dump() == {"prefer": prefer, "decimals": decimals}
    assert saved.decimals == 3 and saved.prefer == "table"


async def test_finalize_context_applies_current_preference_for_every_route() -> None:
    for route in Route:
        ctx = context()
        state = routed_state(ctx, route)
        state.question = "请用文字回答，保留四位小数"
        state.routing_context = RoutingContext(format_preference=state.context.format_preference)
        command = await finalize_context(state, Runtime(context=ctx))
        assert command.update["context"].format_preference == FormatPreferenceContent(prefer="prose", decimals=4)


async def test_table_precision_preserves_claims_and_original_numeric_values() -> None:
    ctx, state, bundle = await synthesis_context(data=True, knowledge=False)
    data = bundle.data.data
    # The exact model view is authoritative, and only the display is rounded.
    draft = data_draft("返回行数为1。", reference=RowCountReference(value=1))
    state.route = RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="结果")
    state.context = TurnContext(time_scope=None, format_preference=FormatPreferenceContent(prefer="table", decimals=2))
    ctx = replace(ctx, llm=FakeChatModel([draft]))
    before = data.model_dump_json()
    answer = (await format_answer(state, Runtime(context=ctx))).update["answer"]
    assert "| 经核验的声明 | 数值展示 |" in answer.markdown
    assert "| 1.00 |" in answer.markdown
    assert answer.claims == draft.claims
    assert data.model_dump_json() == before
    assert len(ctx.llm.calls) == 1


async def test_rounded_nonzero_is_not_rendered_as_exact_zero() -> None:
    ctx = parent_context(Route.DATA_ONLY)
    answer = (await invoke(ctx)).answer
    answer.claims[0].data_refs = [CellReference(row=0, column=0, value="0.0001")]
    answer.format_preference = FormatPreferenceContent(prefer="prose", decimals=2)
    rendered = finalize_answer(answer, await ctx.evidence.read_bundle(ctx.identity))
    assert "≈ 0.00（非精确零）" in rendered.markdown
    assert answer.claims[0].data_refs[0].value == "0.0001"


@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_bad_data_reference_gets_one_repair_then_abstention(repair_succeeds: bool) -> None:
    ctx, state, bundle = await synthesis_context(knowledge=False)
    state.route = RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="数据")
    bad = data_draft("999", value=999)
    ctx = replace(ctx, llm=FakeChatModel([bad, data_draft() if repair_succeeds else bad]))
    answer = (await format_answer(state, Runtime(context=ctx))).update["answer"]
    assert answer.abstained is not repair_succeeds
    assert len(ctx.llm.calls) == 2
    assert "999" not in answer.markdown
    assert answer.evidence_refs == bundle.refs
    validate_formatted_answer(answer, bundle)


async def test_operational_failure_is_not_content_repaired() -> None:
    ctx, state, _ = await synthesis_context(knowledge=False)
    state.route = RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="数据")
    ctx = replace(ctx, llm=FakeChatModel([LlmStructuredOutputError()]))
    command = await format_answer(state, Runtime(context=ctx))
    assert command.update["status"] == "failed"
    assert command.update["failures"][-1].kind is FailureKind.LLM_STRUCTURED_OUTPUT_FAILED
    assert len(ctx.llm.calls) == 1


@pytest.mark.parametrize("version", [1, 2])
async def test_historical_answers_do_not_invent_trace_or_claims(version: int) -> None:
    answer = (await invoke(parent_context(Route.DATA_ONLY))).answer
    raw = answer.model_dump()
    raw["schema_version"] = version
    for field in ("claims", "trace_id", "format_preference", "unanswered", "attempted_sources"):
        raw.pop(field)
    historical = Answer.model_validate(raw)
    assert historical.schema_version == version
    assert historical.markdown == answer.markdown
    assert historical.claims == [] and historical.trace_id is None
    raw["schema_version"] = 3
    with pytest.raises(ValidationError):
        Answer.model_validate(raw)


@pytest.mark.parametrize("field", ["markdown", "confidence", "claims", "citations"])
async def test_commit_guard_rejects_modified_answer(field: str) -> None:
    ctx = parent_context(Route.KNOWLEDGE_ONLY)
    answer = (await invoke(ctx)).answer
    changed = {"markdown": "injected", "confidence": 0, "claims": [], "citations": []}[field]
    with pytest.raises(ConflictError):
        validate_formatted_answer(answer.model_copy(update={field: changed}), await ctx.evidence.read_bundle(ctx.identity))


async def test_formatter_rechecks_passage_ids_at_snapshot_boundary() -> None:
    ctx, state, bundle = await synthesis_context(data=False)
    state.route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="政策")
    citation = Citation.model_validate(bundle.knowledge.knowledge.chunks[0].model_dump(include=set(Citation.model_fields)))
    generated = KnowledgeGeneration(
        passages=(KnowledgePassage(text="声称的政策", chunk_ids=(uuid4(),)),),
        citations=(citation,), attempts=1,
    )
    ctx = replace(ctx, knowledge_generation=AsyncMock(generate=AsyncMock(return_value=generated)))
    command = await format_answer(state, Runtime(context=ctx))
    assert command.update["status"] == "failed"
    assert "answer" not in command.update


async def test_repeated_citations_are_deduplicated_from_snapshot() -> None:
    ctx, state, bundle = await synthesis_context(data=False)
    state.route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="政策")
    citation = Citation.model_validate(bundle.knowledge.knowledge.chunks[0].model_dump(include=set(Citation.model_fields)))
    generated = KnowledgeGeneration(
        passages=(KnowledgePassage(text="【假来源.pdf】规则说明", chunk_ids=(citation.chunk_id,)),
                  KnowledgePassage(text="另一个规则说明", chunk_ids=(citation.chunk_id,))),
        citations=(citation,), attempts=1,
    )
    ctx = replace(ctx, knowledge_generation=AsyncMock(generate=AsyncMock(return_value=generated)))
    answer = (await format_answer(state, Runtime(context=ctx))).update["answer"]
    assert answer.citations == [citation]
    assert len(answer.claims) == 2
    assert "假来源.pdf" not in answer.citations[0].document_title
    validate_formatted_answer(answer, bundle)


async def test_missing_rerank_score_uses_conservative_signal() -> None:
    ctx = parent_context(Route.KNOWLEDGE_ONLY)
    answer = (await invoke(ctx)).answer
    bundle = await ctx.evidence.read_bundle(ctx.identity)
    bundle.knowledge = bundle.knowledge.model_copy(update={
        "knowledge": bundle.knowledge.knowledge.model_copy(update={"top_rerank_score": None})
    })
    assert confidence_for(answer, bundle) == 0.5
