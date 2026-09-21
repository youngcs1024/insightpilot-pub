"""Pure Suite C acceptance is independent of real model quality."""

# ruff: noqa: PLR2004 -- fixed sample sizes and acceptance boundaries.

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.agents.contracts import Route
from app.core.routing import RoutingStrategy
from evals import cli
from evals.harness import routing_cli
from evals.harness.contracts import EvaluationError, Score
from evals.harness.routing import choose, evaluate, metrics, passes
from evals.harness.routing_contracts import CASES, Case, Measurements, Options, Split
from evals.harness.routing_dataset import load_cases
from evals.harness.routing_report import exit_code, markdown, write_report
from tests.eval.routing_support import measurements
from tests.router_support import decision


def test_balanced_reviewed_semantically_disjoint_dataset() -> None:
    cases = load_cases()
    assert len(cases) == 80
    groups = [{c.semantic_group_id for c in cases if c.split is split} for split in Split]
    assert not groups[0] & groups[1]
    for split in Split:
        selected = [c for c in cases if c.split is split]
        assert all(sum(c.expected is route for c in selected) == 10 for route in Route)
        assert sum(c.deliberately_ambiguous for c in selected) >= 10
        assert all(c.rationale and c.reviewed_by == "agent" for c in selected)
    prompt = Path("app/agents/prompts/router.md").read_text()
    assert all(c.question not in prompt for c in cases if c.split is Split.FROZEN)


@pytest.mark.parametrize(
    "defect", ["empty", "duplicate", "group", "quota", "ambiguity", "kind", "yaml_key"]
)
def test_invalid_dataset_fails_before_execution(tmp_path: Path, defect: str) -> None:
    rows = yaml.safe_load(CASES.read_text())
    if defect == "empty":
        rows = []
    elif defect == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif defect == "group":
        rows[-1]["semantic_group_id"] = rows[0]["semantic_group_id"]
    elif defect == "quota":
        rows[0]["expected"] = "both"
    elif defect == "ambiguity":
        for row in rows:
            row["deliberately_ambiguous"] = False
    elif defect == "kind":
        rows[0]["clarification_kind"] = "ambiguous"
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(rows))
    if defect == "yaml_key":
        path.write_text("- id: route-001\n  id: route-002\n")
    with pytest.raises(EvaluationError):
        load_cases(path)


def test_confusion_matrix_shape() -> None:
    report = evaluate(measurements(), load_cases())
    matrix = report.arms[RoutingStrategy.HYBRID].overall.confusion_matrix
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert [matrix[i][i] for i in range(4)] == [30] * 4
    assert report.evidence_valid


def test_misroute_rate_computed() -> None:
    raw = measurements()
    cases = load_cases()
    both = next(c for c in cases if c.expected is Route.BOTH)
    data = next(c for c in cases if c.expected is Route.DATA_ONLY)
    attempts = [a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID]
    next(a for a in attempts if a.case_id == both.id).observation.decision = decision(
        Route.DATA_ONLY
    )
    next(a for a in attempts if a.case_id == data.id).observation.decision = decision(Route.BOTH)
    result = metrics(attempts, cases)
    assert result.misroute == Score(passed=2, total=120)
    assert result.both_misroute == Score(passed=1, total=30)
    assert result.accuracy.passed == 118


def test_per_class_recall_computed() -> None:
    raw = measurements()
    attempts = [a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID]
    attempts[0].observation.decision = None
    result = metrics(attempts, load_cases())
    assert result.per_class[Route.DATA_ONLY].recall == Score(passed=29, total=30)
    assert result.per_class[Route.DATA_ONLY].precision.rate == 1
    assert result.no_decision_by_class[Route.DATA_ONLY] == 1
    assert result.misroute.rate == 0


def test_false_clarification_is_not_specialist_misroute() -> None:
    raw = measurements()
    attempts = [a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID]
    attempts[0].observation.decision = decision(Route.CLARIFY)
    score = metrics(attempts, load_cases())
    assert score.accuracy.rate == 119 / 120
    assert score.misroute.rate == 0
    assert score.clarification_precision == Score(passed=24, total=25)
    assert score.out_of_scope_recall == Score(passed=6, total=6)


def test_prefilter_abstention_is_error_even_on_clarify_truth() -> None:
    report = evaluate(measurements(), load_cases())
    result = report.arms[RoutingStrategy.PREFILTER_ONLY].overall
    assert result.accuracy == Score(passed=0, total=120)
    assert result.clarification_precision.rate is None
    assert result.prefilter_accuracy.rate is None
    assert result.mean_tokens == 0


def test_variance_reported_over_repeats() -> None:
    raw = measurements()
    first = next(a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID and a.repeat == 1)
    first.observation.decision = None
    report = evaluate(raw, load_cases())
    variation = report.arms[RoutingStrategy.HYBRID].variation["accuracy"]
    assert variation.mean == pytest.approx(119 / 120)
    assert variation.sample_stddev == pytest.approx(0.014433756729740657)
    assert len(report.arms[RoutingStrategy.HYBRID].per_repeat) == 3


def test_missing_tokens_remain_unknown() -> None:
    raw = measurements()
    next(a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID).observation.tokens = None
    result = evaluate(raw, load_cases()).arms[RoutingStrategy.HYBRID]
    assert result.overall.mean_tokens is None
    assert result.overall.token_measurements == 119
    assert result.variation["mean_tokens"].mean is None


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "unknown", "dirty", "incomplete", "ids"]
)
def test_incomplete_or_invalid_evidence_never_passes(defect: str) -> None:
    raw = measurements()
    if defect == "missing":
        raw.attempts.pop()
    elif defect == "duplicate":
        raw.attempts[-1] = raw.attempts[0]
    elif defect == "unknown":
        raw.attempts[0].case_id = "unknown"
    elif defect == "dirty":
        raw.config.source_dirty = True
    elif defect == "incomplete":
        raw.complete = False
    else:
        raw.case_ids.append(raw.case_ids[0])
    report = evaluate(raw, load_cases())
    assert not report.evidence_valid
    assert exit_code(report) == 1


def test_threshold_gate_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = measurements()
    for a in raw.attempts:
        if a.arm is RoutingStrategy.HYBRID:
            a.observation.decision = None

    async def collect(options: Options) -> Measurements:
        return raw

    monkeypatch.setattr(routing_cli, "collect", collect)
    assert (
        cli.main(
            ["run", "--suite", "routing", "--report", str(tmp_path), "--threshold-accuracy", "0.90"]
        )
        == 1
    )
    assert (tmp_path / "routing_latest.md").exists()


def test_threshold_equality_and_both_safety_ceiling() -> None:
    metric = evaluate(measurements(), load_cases()).arms[RoutingStrategy.HYBRID].overall
    metric.accuracy = Score(passed=90, total=100)
    metric.both_misroute = Score(passed=5, total=100)
    assert passes(metric)
    metric.both_misroute = Score(passed=6, total=100)
    assert not passes(metric)
    metric.both_misroute = Score(passed=0, total=0)
    assert not passes(metric)


def test_strategy_selection_is_accuracy_first_with_no_misroute_regression() -> None:
    raw = measurements()
    assert choose(evaluate(raw, load_cases())) is RoutingStrategy.HYBRID
    next(a for a in raw.attempts if a.arm is RoutingStrategy.HYBRID).observation.decision = None
    assert choose(evaluate(raw, load_cases())) is RoutingStrategy.LLM_ONLY
    next(
        a for a in raw.attempts if a.arm is RoutingStrategy.LLM_ONLY
    ).observation.decision = decision(Route.BOTH)
    assert choose(evaluate(raw, load_cases())) is RoutingStrategy.HYBRID


def test_report_roundtrip_and_immutable_attempt_file(tmp_path: Path) -> None:
    report = evaluate(measurements(), load_cases())
    write_report(report, tmp_path)
    assert "Confusion matrix" in markdown(report)
    assert "Sample standard deviation" in markdown(report)
    assert "agent-reviewed" in markdown(report)
    assert "offline-model" in markdown(report)
    assert exit_code(report) == 0
    with pytest.raises(FileExistsError):
        write_report(report, tmp_path)


def test_frozen_requires_committed_selection() -> None:
    report = evaluate(measurements(Split.FROZEN), load_cases())
    assert report.evidence_valid
    assert exit_code(report) == 1


def test_cli_routes_defaults_without_changing_nl2sql(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(routing_cli, "run", lambda options: seen.append(options) or 0)
    assert cli.main(["run", "--suite", "routing"]) == 0
    assert seen == [Options()]
    with pytest.raises(SystemExit):
        cli.main(["run", "--suite", "nl2sql", "--split", "frozen"])
    with pytest.raises(SystemExit):
        cli.main(["run", "--suite", "routing", "--threshold-result-accuracy", "0.8"])


@pytest.mark.parametrize("value", [0, 21, -1])
def test_repeats_bounds(value: int) -> None:
    with pytest.raises(ValidationError):
        Options(repeats=value)


def test_invalid_clarification_ground_truth() -> None:
    row = load_cases()[0].model_dump()
    row["expected"] = Route.CLARIFY
    with pytest.raises(EvaluationError):
        Case.model_validate(row)
