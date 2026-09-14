"""The compiled specialist consumes real registered retrieval with HTTP model substitutes."""

from dataclasses import replace

import pytest

from app.retrieval.config import RetrievalConfig
from app.schemas.model_runtime import ModelFailureKind
from model_runtime.errors import ModelError
from tests.agents.knowledge_support import inputs, invoke
from tests.agents.support import context
from tests.retrieval_support import RetrievalHarness, deadline, harness

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["harness"]


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


async def test_real_rerank_outage_returns_bounded_fusion_evidence(harness: RetrievalHarness) -> None:
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
    assert ctx.llm.calls == []
