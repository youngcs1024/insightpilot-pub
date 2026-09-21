"""CLI composition and an explicit development-only strategy selection action."""

import asyncio
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from evals.harness.contracts import EvaluationError
from evals.harness.routing import choose, evaluate, passes
from evals.harness.routing_contracts import (
    MIN_REPEATS,
    SELECTION,
    Options,
    Report,
    Selection,
    Split,
)
from evals.harness.routing_dataset import load_cases
from evals.harness.routing_report import exit_code, write_report
from evals.harness.routing_runtime import collect, digest, snapshot
from scripts.ci_changes import git_bytes
from scripts.dev_route import RouteProcessSettings


def run(options: Options) -> int:
    """Write actual evidence before applying gates; never promote an older latest file."""
    measurements = asyncio.run(collect(options))
    report = evaluate(measurements, load_cases())
    write_report(report, options.report)
    print(f"Report: {options.report / 'routing_latest.md'}")
    return exit_code(report, options.threshold_accuracy)


def select(path: Path) -> int:
    """Freeze a clean development decision, then require its commit before frozen inference."""
    content = path.read_bytes()
    loaded = Report.model_validate_json(content)
    raw = loaded.measurements
    report = evaluate(raw, load_cases())
    current = snapshot(RouteProcessSettings.load())
    if (
        raw.split is not Split.DEVELOPMENT
        or not report.evidence_valid
        or report.identical_arm_accuracy
        or raw.repeats < MIN_REPEATS
        or current.source_dirty
        or raw.config.fingerprint() != current.fingerprint()
    ):
        raise EvaluationError("Selection requires complete current clean development evidence")
    git_bytes(["merge-base", "--is-ancestor", raw.config.git_sha, current.git_sha])
    arm = choose(report)
    if not passes(report.arms[arm].overall):
        raise EvaluationError("Selected strategy does not meet development quality gates")
    if arm is not current.production_strategy:
        raise EvaluationError(
            "Commit the winning production strategy and remeasure development first"
        )
    selection = Selection(
        arm=arm,
        development_run_id=raw.run_id,
        development_sha=raw.config.git_sha,
        development_report_hash=digest(content),
        fingerprint=raw.config.fingerprint(),
    )
    SELECTION.write_text(
        yaml.safe_dump(selection.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    print("Development strategy saved; commit it before evaluating frozen cases.")
    return 0
