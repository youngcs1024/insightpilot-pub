# ruff: noqa: PLR2004 -- explicit metric oracles and acceptance quotas.
"""Failure-aware grid scoring, development-only selection and precision provenance."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.core.errors import RetrievalUnavailableError
from app.schemas.model_runtime import ModelFailureKind
from app.schemas.retrieval import RetrievalTimings
from evals.harness.ablation import choose, evaluate, percentiles, tune_filter
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval import observe, replay_fp32, verify_precision
from evals.harness.retrieval_contracts import Arm, Split
from evals.harness.retrieval_report import exit_code, markdown, write_report
from tests.retrieval_eval_support import attempt, dataset, measurements


def test_complete_grid_and_precision_report(tmp_path: Path) -> None:
    report = evaluate(measurements(), dataset())
    assert report.evidence_valid
    assert len(report.summaries) == 9
    assert all(row.valid == row.attempted == 1 for row in report.summaries)
    assert report.summaries[0].ranking.ndcg_10 < report.summaries[1].ranking.ndcg_10
    text = markdown(report)
    assert "replay" in text
    assert "not human-reviewed" in text
    assert "ΔnDCG" in text
    path = tmp_path / "report.md"
    write_report(report, path)
    assert path.exists()
    assert path.with_suffix(".json").exists()
    assert exit_code(report, 0.1) == 0


def test_failed_missing_duplicate_and_dirty_never_pass() -> None:
    for damage in ("missing", "duplicate", "failure", "dirty", "fallback"):
        raw = measurements()
        if damage == "missing":
            raw.attempts.pop()
        elif damage == "duplicate":
            raw.attempts.append(raw.attempts[0].model_copy(deep=True))
        elif damage == "failure":
            raw.attempts[0].failure_code = "MODEL_UNAVAILABLE"
        elif damage == "dirty":
            raw.source_dirty = True
        else:
            raw.attempts[1].observed.result.degradation = ModelFailureKind.UNAVAILABLE
        report = evaluate(raw, dataset())
        assert not report.evidence_valid
        assert exit_code(report, None) == 2
        assert "No accepted" in markdown(report)


def test_frozen_requires_committed_selection() -> None:
    development = evaluate(measurements(control=False), dataset(), require_control=False)
    selected = choose(development, dataset())
    report = evaluate(measurements(split=Split.FROZEN), dataset(Split.FROZEN))
    assert "uncommitted_development_selection" in report.issues
    raw = measurements(split=Split.FROZEN)
    report = evaluate(raw, dataset(Split.FROZEN), selected, selection_commit="b" * 40)
    assert report.evidence_valid
    with pytest.raises(EvaluationError, match="Frozen"):
        choose(report, dataset(Split.FROZEN))


def test_select_on_development_and_threshold_exit() -> None:
    report = evaluate(measurements(control=False), dataset(), require_control=False)
    selected = choose(report, dataset())
    assert selected.arm is Arm.A_RERANK
    assert selected.config.filtering.absolute_floor == 0.3
    assert selected.config.filtering.dynamic_ratio == 0.5
    report.selection = selected
    report.summaries[1].ranking.ndcg_10 = 0.4
    assert exit_code(report, 0.5) == 1
    assert exit_code(report, 0.4) == 0


def test_filter_sweep_refuses_no_partition_leakage() -> None:
    source = dataset()
    before = source.model_dump_json()
    config = tune_filter(Arm.A_RERANK, measurements(control=False).attempts, source)
    assert source.model_dump_json() == before
    assert config.final_k == 8
    invalid = evaluate(measurements(control=False), source, require_control=False)
    invalid.evidence_valid = False
    with pytest.raises(EvaluationError):
        choose(invalid, source)


def test_precision_candidate_and_revision_mismatch() -> None:
    original, control = attempt(Arm.D_RERANK), attempt(Arm.D_FP32)
    verify_precision(original, control)
    for damage in ("candidate", "revision", "threshold", "precision"):
        changed = control.model_copy(deep=True)
        if damage == "candidate":
            changed.observed.trace.admitted.reverse()
        elif damage == "revision":
            changed.observed.trace.rerank_response.metadata.rerank_revision = "b" * 40
        elif damage == "threshold":
            changed.observed.result.retrieval_config.filtering.absolute_floor = 0.4
        else:
            changed.observed.trace.rerank_response.metadata.precision = "fp16"
        with pytest.raises(EvaluationError):
            verify_precision(original, changed)


def test_corrupt_text_ranking_and_workload_fail() -> None:
    for damage in ("text", "rank", "score", "batch", "unjudged"):
        raw = measurements()
        row = raw.attempts[1].observed
        if damage == "text":
            row.trace.admitted[0].content = "altered original"
        elif damage == "rank":
            row.trace.ranked.reverse()
        elif damage == "score":
            row.trace.rerank_response.scores[0] = 0.05
        elif damage == "batch":
            row.trace.rerank_response.metadata.rerank_batch = 8
        else:
            source = dataset()
            source.labels[0].judgments.pop()
            assert not evaluate(raw, source).evidence_valid
            continue
        assert not evaluate(raw, dataset()).evidence_valid


def test_latency_uses_nearest_rank_and_actual_count() -> None:
    result = percentiles(list(range(1, 21)))
    assert (result.count, result.p50, result.p95) == (20, 10.5, 19)
    assert percentiles([]).p95 is None


async def test_adapter_keeps_failure_without_oracle_input() -> None:
    pipeline = AsyncMock()
    pipeline.retrieve_observed.side_effect = RetrievalUnavailableError(operation="test")
    result = await observe(dataset().cases[0], pipeline, Arm.A)
    assert result.failure_code
    request = pipeline.retrieve_observed.call_args.args[0]
    assert set(type(request).model_fields) == {
        "schema_version",
        "standalone",
        "time_scope",
        "assumptions",
    }


async def test_fp32_replay_calls_only_reranker_with_identical_candidates() -> None:
    original = attempt(Arm.D_RERANK)
    expected = attempt(Arm.D_FP32)
    model = AsyncMock()
    model.rerank.return_value = expected.observed.trace.rerank_response
    result = await replay_fp32(original, model)
    verify_precision(original, result)
    model.embed.assert_not_called()
    assert model.rerank.call_args.args[1] == [
        item.content for item in original.observed.trace.admitted
    ]
    assert result.observed.result.timings.encode_ms == 0
    assert original.observed.trace.rerank_response.metadata.precision == "fp16"


def test_frozen_measurement_rejects_selection_replaced_after_collection() -> None:
    raw = measurements(split=Split.FROZEN)
    selected = choose(evaluate(measurements(), dataset()), dataset())
    replaced = selected.model_copy(update={"rationale": "changed after frozen collection"})
    report = evaluate(raw, dataset(Split.FROZEN), replaced, selection_commit="b" * 40)
    assert "uncommitted_development_selection" in report.issues
    report = evaluate(raw, dataset(Split.FROZEN), selected, selection_commit="c" * 40)
    assert "uncommitted_development_selection" in report.issues


def test_missing_client_stage_measurement_is_not_a_zero_time_success() -> None:
    raw = measurements()
    raw.attempts[0].observed.result.timings = RetrievalTimings(total_ms=42)
    report = evaluate(raw, dataset())
    assert not report.evidence_valid
    assert report.summaries[0].latency_ms["total_ms"].count == 0
