"""Search configuration, safe dates and SDK score contracts without external services."""

import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from pymilvus.exceptions import ParamError

from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.retrieval.config import RetrievalConfig
from app.retrieval.fusion import OUTPUT_FIELDS, SearchArm, arm_request, attach_scores, candidates, time_filter
from app.schemas.retrieval import PolicyPeriod, RangeTimeScope
from tests.retrieval_support import deadline, encoded, fake_store, query, raw_hit


@pytest.mark.parametrize("changes", [
    {"use_dense": False, "use_sparse_learned": False, "use_bm25": False},
    {"pool": 0}, {"pool": 101}, {"pool": 65}, {"rrf_k": 0},
    {"drop_ratio_search": 1.0}, {"hnsw_ef": 0},
])
def test_invalid_search_config_fails_before_io(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(**changes)


def test_rrf_k_is_explicit_in_config() -> None:
    assert RetrievalConfig().model_dump()["rrf_k"] == 60
    assert not RetrievalConfig().record_arm_scores


def test_interval_normalization_preserves_original_labels() -> None:
    scope = RangeTimeScope(periods=[
        PolicyPeriod(start=date(2026, 8, 1), end=date(2026, 9, 1), label="八月"),
        PolicyPeriod(start=date(2026, 7, 1), end=date(2026, 8, 1), label="七月"),
        PolicyPeriod(start=date(2026, 7, 15), end=date(2026, 8, 15), label="重叠"),
    ])
    expression = time_filter(scope)
    assert expression == "((effective_from == -1 or effective_from < 20697) and (effective_to == -1 or effective_to > 20635))"
    assert [period.label for period in scope.periods] == ["八月", "七月", "重叠"]


def test_disjoint_periods_are_parenthesized_or() -> None:
    scope = RangeTimeScope(periods=[
        PolicyPeriod(start=date(2026, 7, 1), end=date(2026, 8, 1)),
        PolicyPeriod(start=date(2026, 9, 1), end=date(2026, 10, 1)),
    ])
    assert ")) or ((" in time_filter(scope)


@pytest.mark.parametrize("start,end", [("2026-08-01", "2026-08-01"), ("2026-09-01", "2026-08-01"), ("bad", "2026-08-01")])
def test_invalid_period_cannot_become_unbounded(start: str, end: str) -> None:
    with pytest.raises(ValidationError):
        PolicyPeriod(start=start, end=end)


async def test_single_arm_reuses_native_score_without_fusion() -> None:
    native = AsyncMock(return_value=[[raw_hit()]])
    hybrid = AsyncMock()
    config = RetrievalConfig(use_sparse_learned=False, use_bm25=False, record_arm_scores=True)
    result = await fake_store(search=native, hybrid_search=hybrid).hybrid_search(
        encoded(), config, query().time_scope, deadline=deadline(), timeout_s=10,
    )
    assert result[0].scores.dense == 0.8
    assert result[0].scores.rrf is None
    native.assert_awaited_once()
    hybrid.assert_not_called()
    assert native.call_args.kwargs["search_params"]["params"]["ef"] == config.hnsw_ef
    assert native.call_args.kwargs["output_fields"] == OUTPUT_FIELDS


async def test_all_arms_use_identical_time_scope_and_record_scores() -> None:
    hit = raw_hit()
    native = AsyncMock(return_value=[[hit]])
    hybrid = AsyncMock(return_value=[[hit]])
    config = RetrievalConfig(record_arm_scores=True, rrf_k=17)
    result = await fake_store(search=native, hybrid_search=hybrid).hybrid_search(
        encoded(), config, query().time_scope, deadline=deadline(), timeout_s=10,
    )
    args = hybrid.call_args.kwargs
    assert args["ranker"].dict()["params"]["k"] == config.rrf_k
    assert all(req.expr == time_filter(query().time_scope) for req in args["reqs"])
    assert all(call.kwargs["filter"] == args["reqs"][0].expr for call in native.call_args_list)
    assert args["reqs"][1].param["params"]["drop_ratio_search"] == config.drop_ratio_search
    assert args["reqs"][2].data == [query().standalone]
    assert args["reqs"][2].param == {"metric_type": "BM25"}
    assert native.await_count == len(SearchArm)
    assert result[0].scores.model_dump(exclude={"schema_version", "rerank"}) == {"dense": 0.8, "sparse_learned": 0.8, "sparse_bm25": 0.8, "rrf": 0.8}
    assert args["timeout"] is None and args["retry_times"] == 0


async def test_production_does_not_issue_diagnostic_searches() -> None:
    native, hybrid = AsyncMock(), AsyncMock(return_value=[[raw_hit()]])
    result = await fake_store(search=native, hybrid_search=hybrid).hybrid_search(
        encoded(), RetrievalConfig(), query().time_scope, deadline=deadline(), timeout_s=10,
    )
    native.assert_not_called()
    assert result[0].scores.rrf == 0.8
    assert result[0].scores.dense is None


@pytest.mark.parametrize("error,expected", [(TimeoutError(), RetrievalUnavailableError), (ParamError(message="bad parameter"), RetrievalConfigurationError)])
async def test_sdk_failure_has_typed_error_and_no_retry(error: Exception, expected: type[Exception]) -> None:
    operation = AsyncMock(side_effect=error)
    with pytest.raises(expected):
        await fake_store(hybrid_search=operation).hybrid_search(encoded(), RetrievalConfig(), query().time_scope, deadline=deadline(), timeout_s=10)
    operation.assert_awaited_once()


async def test_diagnostic_failure_is_not_silently_omitted() -> None:
    with pytest.raises(RetrievalUnavailableError):
        await fake_store(hybrid_search=AsyncMock(return_value=[[raw_hit()]]), search=AsyncMock(side_effect=TimeoutError())).hybrid_search(
            encoded(), RetrievalConfig(record_arm_scores=True), query().time_scope, deadline=deadline(), timeout_s=10,
        )


async def test_cancellation_propagates() -> None:
    with pytest.raises(asyncio.CancelledError):
        await fake_store(hybrid_search=AsyncMock(side_effect=asyncio.CancelledError())).hybrid_search(
            encoded(), RetrievalConfig(), query().time_scope, deadline=deadline(), timeout_s=10,
        )


async def test_expired_budget_does_not_hang() -> None:
    operation = AsyncMock()
    # A pending Future guarantees an await suspension without an artificial sleep.
    async def pending(*args: object, **kwargs: object) -> object:
        return await asyncio.get_running_loop().create_future()
    operation.side_effect = pending
    with pytest.raises(RetrievalUnavailableError):
        await fake_store(hybrid_search=operation).hybrid_search(encoded(), RetrievalConfig(), query().time_scope, deadline=deadline(-1), timeout_s=10)


@pytest.mark.parametrize("raw", [None, [], [[], []], [[{}]]])
def test_malformed_search_batches_fail_closed(raw: object) -> None:
    with pytest.raises(RetrievalUnavailableError):
        candidates(raw, None)


@pytest.mark.parametrize("field,value", [("parent_content", None), ("effective_from", "bad"), ("effective_to", 10**30), ("chunk_uuid", "bad"), ("content_sha256", "bad")])
def test_malformed_entity_failures_are_typed(field: str, value: object) -> None:
    hit = raw_hit()
    hit["entity"][field] = value
    with pytest.raises(RetrievalUnavailableError):
        candidates([[hit]], None)


def test_parent_content_is_populated_and_unbounded_dates_normalize() -> None:
    result = candidates([[raw_hit()]], None)[0]
    assert result.parent_content != result.content
    assert result.effective_from is None and result.effective_to is None


def test_score_join_requires_complete_identity() -> None:
    hit = raw_hit()
    fused = candidates([[hit]], None)
    diagnostics = candidates([[hit]], SearchArm.DENSE)
    diagnostics[0].document_version = "f" * 64
    attach_scores(fused, diagnostics, SearchArm.DENSE)
    assert fused[0].scores.dense is None
    attach_scores(fused, candidates([[hit]], SearchArm.DENSE), SearchArm.DENSE)
    assert fused[0].scores.dense == 0.8


@pytest.mark.parametrize("arm,field", [(SearchArm.DENSE, "dense"), (SearchArm.LEARNED, "sparse")])
def test_enabled_vectors_cannot_be_absent(arm: SearchArm, field: str) -> None:
    value = encoded().model_copy(update={field: None})
    with pytest.raises(RetrievalConfigurationError):
        arm_request(arm, value, RetrievalConfig(), time_filter(query().time_scope))
