"""Combine independently collected evidence without turning partial measurements into a pass."""

import argparse
from pathlib import Path

from pydantic import BaseModel, Field

from spikes.capacity.benchmark import BenchmarkResult
from spikes.capacity.contracts import (
    CapacityResult,
    FailureKind,
    ProbeError,
    Status,
    percentile,
    spearman,
)
from spikes.capacity.observe import Observation, local_verdict
from spikes.capacity.workload import WorkloadResult

COMPARISON_PAIRS = 50
DATABASE_ROWS = 50000
MIN_TURNS = 20


class PrecisionComparison(BaseModel):
    """Numerical feasibility, explicitly not retrieval-quality acceptance."""

    status: Status
    spearman: float
    max_absolute_difference: float
    mean_absolute_difference: float
    fp16_rerank_p50_s: float
    fp16_rerank_p95_s: float
    fp32_rerank_p50_s: float
    fp32_rerank_p95_s: float


class GateReport(BaseModel):
    """Step 0.8 probe gate and later full-application capacity are separate."""

    status: Status = Status.PENDING
    full_application_capacity: Status = Status.PENDING
    local_capacity: CapacityResult | None = None
    precision: PrecisionComparison | None = None
    pending: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class PrerequisiteResult(BaseModel):
    """Explicit execution blockers accompany, but never replace, missing measurements."""

    status: Status
    kind: FailureKind
    detail: str


def compare(fp16: BenchmarkResult, fp32: BenchmarkResult) -> PrecisionComparison:
    """Reject incomplete/different inputs, identities or swapped precision labels."""
    if fp16.readiness is None or fp32.readiness is None:
        raise ProbeError(FailureKind.OUTPUT, "Both precision runs need authenticated readiness.")
    a, b = fp16.readiness.identity, fp32.readiness.identity
    if (
        fp16.status is not Status.PASSED
        or fp32.status is not Status.PASSED
        or fp16.input_sha256 != fp32.input_sha256
        or len(fp16.scores) != COMPARISON_PAIRS
        or len(fp32.scores) != COMPARISON_PAIRS
        or a.precision != "fp16"
        or b.precision != "fp32"
        or a.model_copy(update={"precision": "fp32"}) != b
    ):
        raise ProbeError(FailureKind.OUTPUT, "Precision comparison inputs/identities differ.")
    correlation = spearman(fp16.scores, fp32.scores)
    differences = [abs(x - y) for x, y in zip(fp16.scores, fp32.scores, strict=True)]
    return PrecisionComparison(
        status=Status.PASSED if correlation >= 0.95 else Status.FAILED,  # noqa: PLR2004
        spearman=correlation,
        max_absolute_difference=max(differences),
        mean_absolute_difference=sum(differences) / len(differences),
        fp16_rerank_p50_s=percentile(fp16.timings_s, 0.5),
        fp16_rerank_p95_s=percentile(fp16.timings_s, 0.95),
        fp32_rerank_p50_s=percentile(fp32.timings_s, 0.5),
        fp32_rerank_p95_s=percentile(fp32.timings_s, 0.95),
    )


def read_observations(directory: Path, report: GateReport) -> None:
    """Read explicitly named host measurements; absent GPU samples cannot pass."""
    for filename, remote in (
        ("local-observation.json", False),
        ("server-fp16-observation.json", True),
        ("server-fp32-observation.json", True),
    ):
        path = directory / filename
        if not path.is_file():
            report.pending.append(filename)
            continue
        observation = Observation.model_validate_json(path.read_text())
        if observation.status is not Status.PASSED or not observation.samples:
            report.failures.append(filename)
        if remote and not observation.gpu_samples:
            report.failures.append(filename)
        if not observation.disk_samples:
            report.pending.append(f"{filename}:continuous-disk")
        if not remote:
            report.local_capacity = CapacityResult.model_validate_json(local_verdict(observation))


def read_workloads(directory: Path, report: GateReport) -> None:
    """Validate job receipts and retain pre-exit kernel peaks conservatively."""
    workload_path = directory / "workload.json"
    if workload_path.is_file():
        workload = WorkloadResult.model_validate_json(workload_path.read_text())
        if (
            workload.status is not Status.PASSED
            or workload.inserted_vectors != workload.corpus_size
            or workload.ddl_rows != DATABASE_ROWS
            or len(workload.retrieval_seconds) < MIN_TURNS
            or not workload.cgroup_peak_bytes
        ):
            report.failures.append("workload.json")
        local_path = directory / "local-observation.json"
        if local_path.is_file() and workload.cgroup_peak_bytes:
            # Conservative: retain sampled job usage and add its full kernel peak.
            # This overcounts the job but cannot lose its short pre-exit spike.
            report.local_capacity = CapacityResult.model_validate_json(
                local_verdict(
                    Observation.model_validate_json(local_path.read_text()),
                    workload.cgroup_peak_bytes,
                )
            )
    for filename in ("fp16.json", "fp32.json"):
        path = directory / filename
        if (
            path.is_file()
            and BenchmarkResult.model_validate_json(path.read_text()).status is not Status.PASSED
        ):
            report.failures.append(filename)


def assemble(directory: Path) -> GateReport:
    """Only the complete named evidence set can pass the isolated feasibility gate."""
    report = GateReport()
    prerequisite = directory / "prerequisite.json"
    if (
        prerequisite.is_file()
        and PrerequisiteResult.model_validate_json(prerequisite.read_text()).status is Status.FAILED
    ):
        report.failures.append("prerequisite.json")
    read_observations(directory, report)
    for filename in ("workload.json", "fp16.json", "fp32.json"):
        if not (directory / filename).is_file():
            report.pending.append(filename)
    read_workloads(directory, report)
    if not report.pending and not report.failures:
        report.precision = compare(
            BenchmarkResult.model_validate_json((directory / "fp16.json").read_text()),
            BenchmarkResult.model_validate_json((directory / "fp32.json").read_text()),
        )
        if (
            report.local_capacity is not None
            and report.local_capacity.status is Status.PASSED
            and report.precision.status is Status.PASSED
        ):
            report.status = Status.PASSED
        else:
            report.status = Status.FAILED
    if report.failures:
        report.status = Status.FAILED
    return report


def main() -> int:
    """Print a machine-readable verdict; missing evidence returns nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report = assemble(parser.parse_args().directory)
    print(report.model_dump_json(indent=2))
    return 0 if report.status is Status.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
