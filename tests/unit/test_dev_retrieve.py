"""Diagnostic CLI time, settings precedence and safe error rendering."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.errors import PeriodUnresolved, RetrievalConfigurationError
from app.retrieval.config import RetrievalConfig
from app.retrieval.filtering import diagnostic, filter_ranked
from app.schemas.retrieval import RetrievalResult, RetrievalStage, RetrievalTimings
from scripts import dev_retrieve
from tests.rerank_support import scored
from tests.retrieval_support import query

COMPARISON_PERIODS = 2
NOW = datetime(2026, 8, 1, 20, tzinfo=UTC)


def test_missing_time_disclosed_in_shanghai() -> None:
    result = dev_retrieve.parse_query(dev_retrieve.parser().parse_args(["退款规则"]), NOW)
    assert result.time_scope.as_of.isoformat() == "2026-08-02"
    assert "Asia/Shanghai" in result.assumptions[0]


def test_explicit_historical_scope_is_retained() -> None:
    args = dev_retrieve.parser().parse_args(["退款规则", "--as-of", "2026-07-15"])
    result = dev_retrieve.parse_query(args, NOW)
    assert result.time_scope.as_of.isoformat() == "2026-07-15"
    assert result.assumptions == []


def test_multiple_comparison_periods_keep_labels() -> None:
    args = dev_retrieve.parser().parse_args(
        ["退款规则", "--period", "2026-07-01:2026-08-01", "--period", "2026-08-01:2026-09-01"]
    )
    assert len(dev_retrieve.parse_query(args, NOW).time_scope.periods) == COMPARISON_PERIODS


@pytest.mark.parametrize(
    "flags", [["--as-of", "2026-02-30"], ["--period", "bad"], ["--period", "2026-08-01:2026-07-01"]]
)
def test_invalid_dates_need_clarification(flags: list[str]) -> None:
    with pytest.raises(PeriodUnresolved):
        dev_retrieve.parse_query(dev_retrieve.parser().parse_args(["退款规则", *flags]), NOW)


def test_time_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        dev_retrieve.parser().parse_args(
            ["规则", "--as-of", "2026-08-01", "--period", "2026-07-01:2026-08-01"]
        )


@pytest.mark.parametrize("arms", ["", "dense,dense", "dense,invalid"])
def test_unknown_or_duplicate_arms_rejected(arms: str) -> None:
    args = dev_retrieve.parser().parse_args(["规则", "--arms", arms])
    with pytest.raises(RetrievalConfigurationError):
        dev_retrieve.search_config(args, RetrievalConfig())


def test_diagnostic_defaults_and_explicit_override() -> None:
    args = dev_retrieve.parser().parse_args(["规则", "--arms", "bm25"])
    config = dev_retrieve.search_config(args, RetrievalConfig())
    assert config.record_arm_scores
    assert config.use_bm25
    assert not config.use_dense
    assert not dev_retrieve.search_config(
        args, RetrievalConfig(record_arm_scores=False)
    ).record_arm_scores
    args = dev_retrieve.parser().parse_args(["规则", "--no-record-arm-scores"])
    assert not dev_retrieve.search_config(
        args, RetrievalConfig(record_arm_scores=True)
    ).record_arm_scores


def test_invalid_date_cli_has_safe_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert dev_retrieve.main(["规则", "--as-of", "bad"]) == 1
    assert "PERIOD_UNRESOLVED" in capsys.readouterr().err


def test_bm25_cli_success_uses_process_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = dev_retrieve.RetrievalProcessSettings.model_construct(
        retrieval=dev_retrieve.RetrievalSettings()
    )
    monkeypatch.setattr(dev_retrieve.RetrievalProcessSettings, "load", lambda: settings)
    operation = AsyncMock(return_value=0)
    monkeypatch.setattr(dev_retrieve, "run", operation)
    assert dev_retrieve.main(["规则", "--arms", "bm25"]) == 0
    assert not operation.call_args.args[0].retrieval.search.use_dense


async def test_database_cleanup_survives_model_close_failure() -> None:
    database = SimpleNamespace(aclose=AsyncMock())
    model = SimpleNamespace(aclose=AsyncMock(side_effect=TimeoutError()))
    with pytest.raises(TimeoutError):
        await dev_retrieve.close_resources(database, model)
    database.aclose.assert_awaited_once()


async def test_bm25_cleanup_without_model() -> None:
    database = SimpleNamespace(aclose=AsyncMock())
    await dev_retrieve.close_resources(database, None)
    database.aclose.assert_awaited_once()


def test_rerank_flag_overrides_typed_default() -> None:
    args = dev_retrieve.parser().parse_args(["规则", "--arms", "bm25", "--no-rerank"])
    assert not dev_retrieve.search_config(args, RetrievalConfig()).use_rerank
    args = dev_retrieve.parser().parse_args(["规则", "--rerank"])
    assert dev_retrieve.search_config(args, RetrievalConfig(use_rerank=False)).use_rerank


def test_explain_reaches_cli_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = dev_retrieve.RetrievalProcessSettings.model_construct(
        retrieval=dev_retrieve.RetrievalSettings()
    )
    monkeypatch.setattr(dev_retrieve.RetrievalProcessSettings, "load", lambda: settings)
    operation = AsyncMock(return_value=0)
    monkeypatch.setattr(dev_retrieve, "run", operation)
    assert dev_retrieve.main(["规则", "--explain"]) == 0
    assert operation.call_args.kwargs["explain"] is True


async def test_explain_prints_typed_stage_counts_and_score_ranges(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = SimpleNamespace(start=lambda: None, aclose=AsyncMock())
    model = SimpleNamespace(aclose=AsyncMock())
    config = RetrievalConfig()
    values = scored([0.8, 0.7])
    ranked = filter_ranked(values, config.filtering)
    result = RetrievalResult(
        query=query(),
        corpus_version=None,
        candidates=ranked.candidates,
        retrieval_config=config,
        model_metadata=None,
        timings=RetrievalTimings(),
        reranked=True,
        top_rerank_score=ranked.top_rerank_score,
        meets_floor=ranked.meets_floor,
        stages=[
            diagnostic(RetrievalStage.SEARCH, len(values), values),
            diagnostic(RetrievalStage.ADMISSION, len(values), values),
            diagnostic(RetrievalStage.RERANK, len(values), values),
            *ranked.stages,
        ],
    )

    @asynccontextmanager
    async def store(settings: object) -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr(dev_retrieve, "Database", lambda settings: database)
    monkeypatch.setattr(dev_retrieve, "ModelRuntimeClient", lambda settings: model)
    monkeypatch.setattr(dev_retrieve, "HybridSearchStore", store)
    monkeypatch.setattr(
        dev_retrieve,
        "RetrievalPipeline",
        lambda *args: SimpleNamespace(retrieve=AsyncMock(return_value=result)),
    )
    settings = dev_retrieve.RetrievalProcessSettings.model_construct(
        database=None,
        model_runtime=True,
        retrieval=dev_retrieve.RetrievalSettings(search=config),
    )
    assert await dev_retrieve.run(settings, query(), explain=True) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [item["stage"] for item in printed["stages"]] == list(RetrievalStage)
    assert all(item["input_count"] == len(values) for item in printed["stages"])
    score_range = printed["stages"][-1]["score_ranges"]["rerank"]
    assert score_range == {"schema_version": 1, "minimum": 0.7, "maximum": 0.8}
    database.aclose.assert_awaited_once()
    model.aclose.assert_awaited_once()
