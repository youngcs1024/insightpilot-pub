"""The compiled specialist consumes real registered retrieval with HTTP model substitutes."""

from dataclasses import replace

import pytest

from app.agents.failures import FailureKind
from app.retrieval.config import RetrievalConfig
from app.schemas.model_runtime import ModelFailureKind
from model_runtime.errors import ModelError
from tests.agents.knowledge_support import inputs, invoke
from tests.agents.support import context
from tests.knowledge_query_support import rewrite, topic
from tests.retrieval_support import RetrievalHarness, deadline, harness

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["harness"]
MODEL_ATTEMPTS = 2


async def test_real_pipeline_returns_immutable_evidence(harness: RetrievalHarness) -> None:
    ctx = replace(
        context(responses=[]), retrieval=harness.pipeline(RetrievalConfig()), deadline=deadline()
    )
    output = await invoke(ctx, inputs())
    assert output.evidence.chunks
    assert output.evidence.reranked
    assert output.evidence.meets_floor
    assert output.evidence.corpus_version
    assert all(chunk.document_title == "测试规则" for chunk in output.evidence.chunks)
    assert all(chunk.scores.rerank is not None for chunk in output.evidence.chunks)
    assert len(harness.embeddings.rerank_calls) == 1
    assert ctx.llm.calls == []
    assert ctx.mcp.calls == []


async def test_real_below_floor_abstains(harness: RetrievalHarness) -> None:
    # A floor above the HTTP substitute's real response proves the full filter-to-graph path.
    config = RetrievalConfig()
    config.filtering.absolute_floor = 0.9
    ctx = replace(context(responses=[]), retrieval=harness.pipeline(config), deadline=deadline())
    output = await invoke(ctx)
    assert output.abstained
    assert output.failure is None
    assert output.evidence is None
    assert "阈值=0.9" in output.abstention_reason
    assert "最高重排分数=0.8" in output.abstention_reason
    assert ctx.llm.calls == []


async def test_real_rerank_outage_returns_bounded_fusion_evidence(
    harness: RetrievalHarness,
) -> None:
    harness.embeddings.rerank_error = ModelError()
    config = RetrievalConfig()
    ctx = replace(context(responses=[]), retrieval=harness.pipeline(config), deadline=deadline())
    output = await invoke(ctx)
    assert output.degraded_components == ["rerank"]
    assert output.evidence.degradation is ModelFailureKind.UNAVAILABLE
    assert output.failure is None
    assert not output.abstained
    assert 0 < len(output.evidence.chunks) <= config.filtering.final_k
    assert not output.evidence.reranked
    assert output.evidence.top_rerank_score is None
    assert output.evidence.meets_floor is None
    assert all(chunk.scores.rerank is None for chunk in output.evidence.chunks)
    assert len(harness.embeddings.rerank_calls) == MODEL_ATTEMPTS
    assert ctx.llm.calls == []


async def test_natural_language_comparison_reaches_real_validity_filters(
    harness: RetrievalHarness,
) -> None:
    ctx = replace(
        context(responses=[]), retrieval=harness.pipeline(RetrievalConfig()), deadline=deadline()
    )
    output = await invoke(ctx, inputs(question="比较2026年7月和8月的七天退款规则", time_scope=None))
    assert output.failure is None
    assert output.clarification is None
    assert {chunk.source_path for chunk in output.evidence.chunks} >= {"july.md", "august.md"}
    assert [period.label for period in output.evidence.time_scope.periods] == [
        "2026年7月",
        "2026年8月",
    ]
    assert output.evidence.original_question == "比较2026年7月和8月的七天退款规则"
    assert ctx.llm.calls == []


async def test_real_encoding_outage_is_typed_failure_without_rerank(
    harness: RetrievalHarness,
) -> None:
    before = len(harness.embeddings.calls)
    harness.embeddings.encode_error = ModelError()
    ctx = replace(context(), retrieval=harness.pipeline(RetrievalConfig()), deadline=deadline())
    output = await invoke(ctx)
    assert output.failure.kind is FailureKind.MODEL_RUNTIME_UNAVAILABLE
    assert output.evidence is None
    assert not output.abstained
    assert output.degraded_components == []
    assert len(harness.embeddings.calls) == before + MODEL_ATTEMPTS
    assert harness.embeddings.rerank_calls == []
    assert ctx.llm.calls == []


async def test_followup_inherits_real_august_policy_boundary(harness: RetrievalHarness) -> None:
    prior = topic()
    ctx = replace(
        context(responses=[rewrite(prior, "七天退款规则中的运费")]),
        retrieval=harness.pipeline(RetrievalConfig()),
        deadline=deadline(),
    )
    output = await invoke(
        ctx, inputs(question="那运费呢\uff1f", time_scope=None, knowledge_history=[prior])
    )
    paths = {chunk.source_path for chunk in output.evidence.chunks}
    assert "august.md" in paths
    assert "july.md" not in paths
    assert output.evidence.time_scope.periods[0].start == prior.time_scope.periods[0].start
    assert output.assumptions
