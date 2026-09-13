"""Assemble separately authorized GPU evidence without invoking GPU work implicitly."""

import argparse
from pathlib import Path

from pydantic import AwareDatetime, Field

from app.schemas.mcp import Contract
from app.schemas.model_runtime import ReadyResult
from evals.harness.ablation import evaluate
from evals.harness.retrieval_contracts import Measurements, Selection, Split
from evals.harness.retrieval_dataset import load_dataset
from evals.harness.retrieval_runtime import committed_selection
from scripts.model_evidence import Provenance


class HardwareSample(Contract):
    """Actual shared-device observations, distinct from model allocator peaks."""

    at: AwareDatetime
    total_mib: int = Field(gt=0)
    used_mib: int = Field(ge=0)
    free_mib: int = Field(ge=0)
    other_process_mib: int = Field(ge=0)
    precision: str = Field(pattern=r"^fp(?:16|32)$")


class EnvironmentEvidence(Contract):
    """Operator-owned receipts record deployment, shared workload and restoration."""

    server: Provenance
    samples: list[HardwareSample] = Field(min_length=3)
    restored: ReadyResult


class AcceptanceBundle(Contract):
    """Complete final-source evidence; no cached passed boolean is accepted."""

    client_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    development: Measurements
    frozen: Measurements
    selection: Selection
    selection_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment: EnvironmentEvidence


def issues(bundle: AcceptanceBundle, expected_sha: str) -> list[str]:
    """Recompute full grid acceptance using the checked-in judgments."""
    result: list[str] = []
    if bundle.client_sha != expected_sha or any(
        raw.client_sha != expected_sha or raw.source_dirty
        for raw in (bundle.development, bundle.frozen)
    ):
        result.append("client_revision")
    if not (bundle.development.server == bundle.frozen.server == bundle.environment.server):
        result.append("server_provenance")
    if bundle.development.split is not Split.DEVELOPMENT or bundle.frozen.split is not Split.FROZEN:
        result.append("partition_identity")
    for raw in (bundle.development, bundle.frozen):
        dataset = load_dataset(raw.split)
        report = evaluate(
            raw, dataset, bundle.selection if raw.split is Split.FROZEN else None,
            selection_commit=bundle.selection_commit if raw.split is Split.FROZEN else None,
        )
        result.extend(report.issues)
    environment = bundle.environment
    if environment.restored.metadata.precision != "fp16":
        result.append("fp16_not_restored")
    if {sample.precision for sample in environment.samples} != {"fp16", "fp32"}:
        result.append("missing_shared_load_measurements")
    if any(sample.used_mib > sample.total_mib or sample.free_mib > sample.total_mib for sample in environment.samples):
        result.append("invalid_hardware_sample")
    return result


def main() -> int:
    """Build the private bundle from newly measured artifacts, after FP16 restoration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--client-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection, commit = committed_selection(args.selection)
    bundle = AcceptanceBundle(
        client_sha=args.client_sha,
        development=Measurements.model_validate_json(args.development.read_text()),
        frozen=Measurements.model_validate_json(args.frozen.read_text()),
        selection=selection, selection_commit=commit,
        environment=EnvironmentEvidence.model_validate_json(args.environment.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2))
    failures = issues(bundle, args.client_sha)
    print("GPU retrieval evidence: " + (", ".join(failures) if failures else "complete"))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
