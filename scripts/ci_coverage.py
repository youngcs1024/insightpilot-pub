"""Collect partial coverage for diagnosis while requiring all partitions for acceptance."""

import argparse
from pathlib import Path

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException
from pydantic import ValidationError

from scripts.check_test_coverage import CoverageReport, summarize
from scripts.ci_evidence import CheckEvidence, CoverageState
from scripts.ci_junit import FileState, file_state
from scripts.ci_partitions import PARTITIONS, Partition
from scripts.ci_policy import Result
from scripts.ci_result import emit


def partition_state(path: Path, outcome: Result) -> CoverageState:
    """Missing output is expected only when the upstream partition never ran."""
    state = file_state(path)
    if state is FileState.MISSING and outcome is Result.SKIPPED:
        return CoverageState.EXPECTED_MISSING
    if state is not FileState.PRESENT:
        return CoverageState(state.value)
    try:
        data = CoverageData(basename=str(path))
        data.read()
        return CoverageState.VALID if data.measured_files() else CoverageState.INVALID
    except (CoverageException, OSError, ValueError):
        return CoverageState.INVALID


def available_data(
    root: Path, summary: Path, upstream: dict[Partition, Result]
) -> dict[Partition, CoverageState]:
    """Validate each database independently and preserve typed failure causes."""
    states = {
        name: partition_state(root / f".coverage.{name}", upstream[name]) for name in PARTITIONS
    }
    for name, state in states.items():
        emit(f"Coverage {name}: {state.value}.", summary)
    return states


def collect(
    root: Path,
    summary: Path,
    outcomes: tuple[Result, Result, Result],
    *,
    output: Path | None = None,
) -> bool:
    """Report useful measurements even when the authoritative test job failed."""
    upstream = dict(zip(PARTITIONS, outcomes, strict=True))
    states = available_data(root, summary, upstream)
    available = [
        str(root / f".coverage.{name}")
        for name, state in states.items()
        if state is CoverageState.VALID
    ]
    evidence = CheckEvidence(
        checks_passed=None,
        evidence_valid=len(available) == len(PARTITIONS),
        upstream=upstream,
        partitions=states,
    )
    complete = len(available) == len(PARTITIONS) and all(
        outcome is Result.SUCCESS for outcome in outcomes
    )
    if not complete:
        emit("Coverage: diagnostic only; missing evidence or unsuccessful test partition.", summary)
    if not available:
        return finish(evidence, summary, output)
    try:
        coverage = Coverage(data_file=str(root / ".coverage"))
        coverage.combine(data_paths=available, strict=True, keep=True)
        coverage.save()
        coverage.json_report(outfile=str(root / ".coverage.json"))
        report = CoverageReport.model_validate_json((root / ".coverage.json").read_text())
        results = summarize(report, root)
    except (CoverageException, OSError, ValueError, ValidationError):
        emit("Coverage: invalid combined evidence; no acceptance claimed.", summary)
        evidence.evidence_valid = False
        return finish(evidence, summary, output)
    for result in results:
        emit(
            f"{result.directory}: {result.covered_lines}/{result.num_statements} "
            f"({result.percent:.2f}%); threshold {'PASS' if result.passed else 'FAIL'}.",
            summary,
        )
    evidence.checks_passed = all(result.passed for result in results)
    return finish(evidence, summary, output)


def finish(evidence: CheckEvidence, summary: Path, output: Path | None) -> bool:
    """Persist every terminal outcome, including missing and corrupt input artifacts."""
    if output is not None:
        output.write_text(evidence.model_dump_json(indent=2) + "\n")
    emit(evidence.describe(), summary)
    emit(f"Coverage acceptance: {'PASS' if evidence.accepted else 'FAIL'}.", summary)
    return evidence.accepted


def main() -> None:
    """Use current-run artifacts and raw dependency outcomes supplied by the workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PARTITIONS:
        parser.add_argument("--" + name, type=Result, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        0
        if collect(
            Path.cwd(),
            args.summary,
            (args.unit, args.integration, args.storage),
            output=args.output,
        )
        else 1
    )


if __name__ == "__main__":
    main()
