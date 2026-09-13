"""Typed post-search orchestration with no real service or storage dependency."""
# ruff: noqa: PLR2004 -- exact batched workload and typed failure expectations.

import asyncio
import time

import pytest

from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, RetrievalConfigurationError
from app.retrieval.config import RetrievalConfig
from app.retrieval.reranking import rerank_candidates
from app.schemas.retrieval import StageStatus
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelError,
    ModelInputError,
    ModelOOMError,
    ModelQueueError,
)
from tests.rerank_support import model, scored
from tests.retrieval_support import deadline


async def test_rerank_is_single_batched_child_call_and_scores_remain_attached() -> None:
    values = scored([0.2] * 20)
    before = [item.model_dump() for item in values]
    client = model([0.8] * 19 + [0.9])
    result = await rerank_candidates("规则", values, client, RetrievalConfig(), deadline=deadline())
    client.rerank.assert_awaited_once_with(
        "规则",
        [item.content for item in values],
        deadline=client.rerank.call_args.kwargs["deadline"],
        max_length=320,
    )
    assert result.reranked
    assert result.meets_floor
    assert result.top_rerank_score == 0.9
    first = result.candidates[0]
    assert first.chunk_uuid == values[-1].chunk_uuid
    assert first.parent_content == values[-1].parent_content
    assert first.parent_content != first.content
    assert first.scores.dense == values[-1].scores.dense
    assert [item.model_dump() for item in values] == before
    assert result.candidates[1].chunk_uuid == values[0].chunk_uuid


@pytest.mark.parametrize(
    "failure", [ModelError(), ModelQueueError(), ModelOOMError(), ModelDeadlineError()]
)
async def test_availability_failures_degrade_with_unknown_relevance(failure: ModelError) -> None:
    values = scored([0.0] * 10, ["a.md"] * 5 + ["b.md"] * 5)
    for item in values:
        item.scores.rerank = None
    client = model([])
    client.rerank.side_effect = failure
    result = await rerank_candidates("规则", values, client, RetrievalConfig(), deadline=deadline())
    assert not result.reranked
    assert result.degradation == failure.kind
    assert result.top_rerank_score is None
    assert result.meets_floor is None
    assert result.candidates == [*values[:3], *values[5:8]]
    assert result.stages[0].status is StageStatus.DEGRADED
    assert all(item.scores.rerank is None for item in result.candidates)
    client.rerank.assert_awaited_once()


@pytest.mark.parametrize("failure", [ModelAuthError(), ModelInputError(), ModelContractError()])
async def test_non_availability_failures_are_not_disguised(failure: ModelError) -> None:
    client = model([])
    client.rerank.side_effect = failure
    with pytest.raises(type(failure)):
        await rerank_candidates(
            "规则", scored([0.8]), client, RetrievalConfig(), deadline=deadline()
        )


async def test_empty_pool_makes_no_model_call() -> None:
    client = model([])
    result = await rerank_candidates("规则", [], client, RetrievalConfig(), deadline=deadline())
    assert result.candidates == []
    assert result.degradation is None
    assert result.stages[0].status is StageStatus.EMPTY
    client.rerank.assert_not_called()


async def test_explicit_disable_returns_unfiltered_pool_without_model() -> None:
    values = scored([0.1] * 20, ["same.md"] * 20)
    result = await rerank_candidates(
        "规则", values, None, RetrievalConfig(use_rerank=False), deadline=deadline()
    )
    assert result.candidates == values
    assert result.degradation is None
    assert result.meets_floor is None
    assert result.stages[0].status is StageStatus.DISABLED


async def test_enabled_reranking_requires_a_model() -> None:
    with pytest.raises(RetrievalConfigurationError):
        await rerank_candidates("规则", scored([0.8]), None, RetrievalConfig(), deadline=deadline())


async def test_cancellation_propagates() -> None:
    client = model([])
    client.rerank.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await rerank_candidates(
            "规则", scored([0.8]), client, RetrievalConfig(), deadline=deadline()
        )


async def test_expired_total_deadline_cannot_become_fallback() -> None:
    client = model([])
    budget = deadline()

    async def exhausted(*args: object, **kwargs: object) -> None:
        object.__setattr__(budget, "at", time.monotonic() - 1)
        raise ModelDeadlineError()

    client.rerank.side_effect = exhausted
    with pytest.raises(DeadlineExceededError):
        await rerank_candidates("规则", scored([0.8]), client, RetrievalConfig(), deadline=budget)


async def test_initial_expired_deadline_skips_models() -> None:
    client = model([])
    with pytest.raises(DeadlineExceededError):
        await rerank_candidates(
            "规则", scored([0.8]), client, RetrievalConfig(), deadline=Deadline(0)
        )
    client.rerank.assert_not_called()
