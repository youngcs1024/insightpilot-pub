"""Precision evidence must match identities and handle ties without invented scores."""

import pytest

from app.core.errors import ValidationError
from scripts.bench_model_runtime import Benchmark
from scripts.compare_model_precision import compare, ranks
from tests.fakes.model_runtime import FakeModels


def artifact(precision: str) -> Benchmark:
    metadata = FakeModels().metadata().model_copy(update={"precision": precision})
    return Benchmark(
        input_sha256="a" * 64,
        metadata=metadata,
        scores=[index / 50 for index in range(50)],
        client_seconds=[1.0] * 50,
        server_ms=[100] * 50,
        queue_ms=[0] * 50,
        inference_ms=[100] * 50,
        p50_s=1,
        p95_s=1,
        relevant_above_control=True,
        accepted=True,
    )


def test_identical_order_has_perfect_correlation_and_ties_use_average_ranks() -> None:
    assert ranks([5, 2, 2]) == [2, 0.5, 0.5]
    result = compare(artifact("fp16"), artifact("fp32"))
    assert result.passed
    assert result.spearman == pytest.approx(1)


def test_incompatible_or_constant_evidence_cannot_pass() -> None:
    other = artifact("fp32")
    other.input_sha256 = "b" * 64
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), other)
    other = artifact("fp32")
    other.scores = [0.5] * 50
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), other)
