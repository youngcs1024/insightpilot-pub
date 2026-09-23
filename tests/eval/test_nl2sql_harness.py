"""Pure harness acceptance; live model quality is a separate explicit run."""

# ruff: noqa: PLR2004 -- numerical comparator boundaries and fixed v1 acceptance quotas.

import asyncio
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core.deadline import Deadline
from app.core.errors import McpUnavailableError, SqlExecutionError
from app.schemas.mcp import SqlErrorKind
from app.services.metric_binding import build_binding
from evals import cli
from evals.harness.adversarial import evaluate as evaluate_adversarial
from evals.harness.adversarial import exit_code as adversarial_exit_code
from evals.harness.adversarial import load_adversaries
from evals.harness.compare import compare
from evals.harness.contracts import Comparison, EvaluationError, Observation, Options
from evals.harness.dataset import load_cases
from evals.harness.nl2sql import bindings_match, observe
from evals.harness.report import exit_code, markdown, write_report
from evals.harness.runner import evaluate, run_suite
from tests.agents.correction_support import correction_context
from tests.agents.support import context, metric_intent, sql_candidate
from tests.eval.support import config, payload
from tests.metric_resolution_support import request, schema


def test_scalar_comparison_respects_tolerance() -> None:
    assert compare(payload([[100.09]]), payload([[100]]), Comparison.SCALAR)
    assert not compare(payload([[100.11]]), payload([[100]]), Comparison.SCALAR)
    assert not compare(payload([[0.000001]]), payload([[0]]), Comparison.SCALAR)
    assert compare(payload([[0]]), payload([[0]]), Comparison.SCALAR)
    assert compare(
        payload([["123456789123456789.123"]]),
        payload([["123456789123456789.123"]]),
        Comparison.SCALAR,
        0,
    )
    assert not compare(payload([[True]], ["bool"]), payload([[1]]), Comparison.SCALAR)
    assert not compare(payload([["1"]], ["text"]), payload([[1]]), Comparison.SCALAR)


def test_set_comparison_ignores_row_order() -> None:
    assert compare(payload([[1], [2], [1]]), payload([[1], [1], [2]]), Comparison.SET)
    assert not compare(payload([[1], [2], [2]]), payload([[1], [1], [2]]), Comparison.SET)


def test_set_comparison_ignores_column_alias() -> None:
    actual = payload([["1.00"]])
    expected = payload([[1]], ["int8"])
    actual.columns[0].name = "renamed_total"
    assert compare(actual, expected, Comparison.SET)
    assert not compare(payload([["1"]], ["text"]), expected, Comparison.SET)


def test_ordered_comparison_respects_order() -> None:
    assert not compare(payload([[1], [2]]), payload([[2], [1]]), Comparison.ORDERED)
    assert compare(payload([[1], [2]]), payload([[1], [2]]), Comparison.ORDERED)


@pytest.mark.parametrize("mode", list(Comparison))
def test_truncated_results_never_pass(mode: Comparison) -> None:
    actual = payload([[1]])
    actual.result_truncated = True
    assert not compare(actual, payload([[1]]), mode)
    assert not compare(payload([[1]]), actual, mode)


def test_null_empty_and_shape_are_distinct() -> None:
    assert compare(payload([]), payload([]), Comparison.EMPTY)
    assert not compare(payload([[None]]), payload([]), Comparison.EMPTY)
    assert not compare(payload([]), payload([[None]]), Comparison.SCALAR)
    assert compare(payload([[None]]), payload([[None]]), Comparison.SCALAR)
    assert not compare(payload([[0]]), payload([[None]]), Comparison.SCALAR)
    assert not compare(payload([[1, 2]], ["int8", "int8"]), payload([[1]]), Comparison.SET)


def test_complete_per_metric_bindings_required() -> None:
    case = load_cases()[0]
    binding = build_binding(request(), schema()).binding
    observation = Observation(bindings=[binding])
    assert bindings_match(case, observation)
    observation.bindings[0].filters_applied.append("o.region_id IN (3)")
    assert not bindings_match(case, observation)
    observation.bindings[0].filters_applied = []
    assert not bindings_match(case, observation)
    assert not bindings_match(case, Observation())


def test_binding_period_metric_and_date_field_mismatch() -> None:
    case = load_cases()[0]
    binding = build_binding(request(), schema()).binding
    for update in (
        {"metric_key": "aov"},
        {"period_end": binding.period_end + timedelta(days=1)},
        {"date_field": "o.created_at"},
    ):
        assert not bindings_match(case, Observation(bindings=[binding.model_copy(update=update)]))
    assert not bindings_match(case, Observation(bindings=[binding, binding]))


async def test_adversarial_sql_never_reaches_model_or_database() -> None:
    generate, execute = AsyncMock(), AsyncMock()
    adversaries = [case for case in load_cases() if case.adversarial_sql]
    for case in adversaries:
        result = await evaluate(case, 1, generate, execute)
        assert result.unsafe_sql_blocked
        assert result.evidence_valid
    generate.assert_not_awaited()
    execute.assert_not_awaited()


async def test_failed_attempts_remain_in_denominator_and_repeats(tmp_path: Path) -> None:
    cases = [load_cases()[0], load_cases()[-1]]
    generate = AsyncMock(side_effect=McpUnavailableError())
    execute = AsyncMock(return_value=payload([[42]]))
    report = await run_suite(cases, Options(repeats=3), config(), generate, execute)
    assert report.summary.execution_match.total == 3
    assert report.summary.result_accuracy.passed == 0
    assert report.summary.unsafe_sql_block_rate.passed == 3
    assert len(report.results) == report.expected_attempts == 6
    assert list(report.per_repeat) == [1, 2, 3]
    assert report.per_trap["T1"].result_accuracy.total == 3
    assert generate.call_count == execute.call_count == 3
    assert generate.call_args.args == (cases[0].question,)
    write_report(report, tmp_path)
    assert (tmp_path / "nl2sql_latest.json").read_text() == (
        tmp_path / f"nl2sql_{report.run_id}.json"
    ).read_text()


async def test_report_includes_config_snapshot() -> None:
    report = await run_suite(
        load_cases()[:1],
        Options(),
        config(),
        AsyncMock(return_value=Observation(result=payload([[42]]))),
        AsyncMock(return_value=payload([[42]])),
    )
    text = markdown(report)
    for value in ("scripted-test-model", "sql_generate.md", "catalog_versions", "git_sha", "T1"):
        assert value in text
    assert "test-key" not in text
    assert report.summary.result_accuracy.rate == 1
    assert exit_code(report, 0.8) == 1  # missing safety denominator is not a pass


async def test_threshold_exits_nonzero_below_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = await run_suite(
        [load_cases()[0], *[c for c in load_cases() if c.adversarial_sql]],
        Options(),
        config(),
        AsyncMock(return_value=Observation(result=payload([[0]]))),
        AsyncMock(return_value=payload([[42]])),
    )
    monkeypatch.setattr(cli, "run", AsyncMock(return_value=report))
    # Invoke synchronous main outside an already running event loop.
    code = await asyncio.to_thread(
        cli.main, ["run", "--report", str(tmp_path), "--threshold-result-accuracy", "0.8"]
    )
    assert code == 1
    assert (tmp_path / "nl2sql_latest.md").exists()
    assert exit_code(report, 0) == 0
    assert exit_code(report.model_copy(update={"expected_attempts": 3}), 0) == 1
    report.config.source_dirty = True
    assert exit_code(report, 0) == 1


async def test_canonical_failure_invalidates_evidence_without_generation() -> None:
    generate = AsyncMock()
    result = await evaluate(
        load_cases()[0], 1, generate, AsyncMock(side_effect=McpUnavailableError())
    )
    assert not result.evidence_valid
    generate.assert_not_awaited()
    truncated = payload([[42]])
    truncated.result_truncated = True
    result = await evaluate(load_cases()[0], 1, generate, AsyncMock(return_value=truncated))
    assert not result.evidence_valid
    generate.assert_not_awaited()


async def test_anchor_mismatch_is_not_a_model_failure() -> None:
    case = next(c for c in load_cases() if c.id == "nl2sql-test-account-001")
    result = await evaluate(case, 1, AsyncMock(), AsyncMock(return_value=payload([[1800]])))
    assert not result.evidence_valid
    assert result.failure_code == "canonical_anchor_mismatch"


async def test_graph_adapter_retains_full_result_and_bindings() -> None:
    rows = [[n] for n in range(250)]
    ctx = context(responses=[metric_intent(), sql_candidate()], mcp_results=[payload(rows)])
    result = await observe("2026年8月的GMV", ctx)
    assert result.result.rows == rows
    assert result.bindings[0].metric_key == "gmv"
    assert result.corrections == 0
    assert ctx.evidence.snapshot is None


async def test_graph_adapter_retains_failed_sql_semantics_and_corrections() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_COLUMN)] * 3)
    result = await observe("2026年8月的GMV", ctx)
    assert result.result is None
    assert result.bindings
    assert result.corrections == 2
    assert result.failure_code is not None


async def test_graph_deadline_does_not_consume_model_queue() -> None:
    ctx = replace(context(), deadline=Deadline(0))
    result = await observe("GMV", ctx)
    assert result.failure_code is not None
    assert not ctx.llm.calls


def test_dataset_has_complete_fixed_v1_coverage() -> None:
    cases = load_cases()
    assert len(cases) == 48
    ordinary = [c for c in cases if not c.adversarial_sql]
    assert len(ordinary) == 26
    assert sum(bool(c.adversarial_sql) for c in cases) == 22
    assert Counter(trap for c in ordinary for trap in c.traps) == dict.fromkeys(
        [f"T{n}" for n in range(1, 9)], 3
    )
    assert {c.comparison for c in ordinary} == set(Comparison)
    assert sum(c.comparison is Comparison.EMPTY for c in ordinary) == 2


def test_offline_adversarial_gate_covers_all_rejections_and_rewrites(tmp_path: Path) -> None:
    cases = load_adversaries()
    report = evaluate_adversarial(cases)
    assert report.blocked.passed == report.blocked.total == 22
    assert report.rewrites.passed == report.rewrites.total == 2
    assert adversarial_exit_code(report, 1.0) == 0
    assert cli.main(
        ["run", "--suite", "adversarial", "--threshold-block-rate", "1.0", "--report", str(tmp_path)]
    ) == 0
    assert len(list(tmp_path.glob("adversarial_*.json"))) == 1
    failed = report.model_copy(deep=True)
    failed.results[0].passed = False
    failed.blocked.passed -= 1
    assert adversarial_exit_code(failed, 1.0) == 1
    failed = report.model_copy(deep=True)
    failed.rewrites.passed -= 1
    assert adversarial_exit_code(failed, 1.0) == 1


@pytest.mark.parametrize("content", ["[]", "- id: nl2sql-unsafe-001\n  sql: SELECT 1", "["])
def test_incomplete_or_malformed_adversarial_dataset_fails_closed(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "adversarial.yaml"
    path.write_text(content)
    with pytest.raises(EvaluationError):
        load_adversaries(path)


@pytest.mark.parametrize("content", ["[]", "[invalid]", "x: [", "- id: nl2sql-x\n  question: x"])
def test_bad_static_dataset_fails_closed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(content)
    with pytest.raises(EvaluationError):
        load_cases(path)


@pytest.mark.parametrize(
    "values", [{"repeats": 0}, {"threshold_result_accuracy": 1.1}, {"repeats": 21}]
)
def test_invalid_options_rejected(values: dict[str, int | float]) -> None:
    with pytest.raises(ValidationError):
        Options.model_validate(values)


def test_decimal_comparison_does_not_round_to_context_precision() -> None:
    left = payload([["123456789123456789123456789123456789.1"]])
    right = payload([["123456789123456789123456789123456789.2"]])
    assert not compare(left, right, Comparison.SET)
    assert not compare(left, right, Comparison.SCALAR, 0)
    assert not compare(payload([[None]], ["text"]), payload([[None]]), Comparison.SET)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("- id: nl2sql-x\n  id: nl2sql-y\n")
    with pytest.raises(EvaluationError):
        load_cases(path)


async def test_semantics_are_measured_even_when_sql_fails() -> None:
    binding = build_binding(request(), schema()).binding
    observation = Observation(bindings=[binding], failure_code="SQL_EXECUTION_FAILED")
    result = await evaluate(
        load_cases()[0],
        1,
        AsyncMock(return_value=observation),
        AsyncMock(return_value=payload([[42]])),
    )
    assert result.metric_resolution_accuracy
    assert not result.execution_match
    assert not result.result_accuracy


def test_invalid_ground_truth_filter_is_dataset_error(tmp_path: Path) -> None:
    case = load_cases()[0]
    case.expected_bindings[0].filters = ["SELECT * FROM biz.orders"]
    path = tmp_path / "cases.yaml"
    # JSON is a YAML subset and avoids relying on another YAML writer in tests.
    path.write_text("[" + case.model_dump_json() + "]")
    with pytest.raises(EvaluationError):
        load_cases(path)
