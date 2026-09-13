"""Dedicated evidence validator is tested with explicit synthetic artifacts in ordinary CI."""

from datetime import UTC, datetime

import pytest

from app.schemas.model_runtime import ReadyResult
from evals.harness import retrieval_acceptance
from evals.harness.ablation import choose, evaluate
from evals.harness.retrieval_acceptance import AcceptanceBundle, EnvironmentEvidence, HardwareSample
from tests.ingestion_support import model_metadata
from tests.retrieval_eval_support import SHA, dataset, measurements
from evals.harness.retrieval_contracts import Split


def bundle() -> AcceptanceBundle:
    selected = choose(evaluate(measurements(control=False), dataset(), require_control=False), dataset())
    return AcceptanceBundle(
        client_sha=SHA, development=measurements(), frozen=measurements(split=Split.FROZEN),
        selection=selected, selection_commit="b" * 40,
        environment=EnvironmentEvidence(
            server=measurements().server,
            restored=ReadyResult(request_id="scripted", metadata=model_metadata()),
            samples=[HardwareSample(at=datetime.now(UTC), total_mib=24000, used_mib=5000, free_mib=19000, other_process_mib=0, precision=precision) for precision in ("fp16", "fp32", "fp16")],
        ),
    )


def test_gpu_bundle_recomputes_current_source_and_restoration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_acceptance, "load_dataset", dataset)
    artifact = bundle()
    assert not retrieval_acceptance.issues(artifact, SHA)
    artifact.environment.restored.metadata.precision = "fp32"
    assert "fp16_not_restored" in retrieval_acceptance.issues(artifact, SHA)
    assert "client_revision" in retrieval_acceptance.issues(bundle(), "c" * 40)


def test_gpu_bundle_rejects_missing_control_and_load_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_acceptance, "load_dataset", dataset)
    artifact = bundle()
    artifact.development.attempts.pop()
    for sample in artifact.environment.samples:
        sample.precision = "fp16"
    result = retrieval_acceptance.issues(artifact, SHA)
    assert "missing_precision_control" in result
    assert "missing_shared_load_measurements" in result
