"""Precision evidence must match identities and handle ties without invented scores."""
# ruff: noqa: PLR2004 -- fixed acceptance workload and synthetic measurements.

import json

import pytest
from pydantic import ValidationError as SchemaValidationError

from app.core.errors import ValidationError
from scripts.compare_model_precision import compare, ranks
from scripts.model_evidence import Benchmark, CallMeasurement, Provenance, RerankMeasurement
from tests.fakes.model_runtime import FakeModels


def artifact(precision: str) -> Benchmark:
    metadata = FakeModels().metadata().model_copy(update={"precision": precision})
    call = CallMeasurement(
        request_id="synthetic-request",
        metadata=metadata,
        client_seconds=1,
        server_ms=100,
        queue_ms=10,
        inference_ms=90,
    )
    return Benchmark(
        provenance=Provenance(
            source_sha="a" * 40,
            image_id="sha256:" + "b" * 64,
            gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        ),
        input_sha256="a" * 64,
        metadata=metadata,
        embedding=call.model_copy(deep=True),
        relevance=RerankMeasurement(call=call.model_copy(deep=True), scores=[0.9, 0.1]),
        precision_calls=[
            RerankMeasurement(
                call=call.model_copy(deep=True),
                scores=[index / 50 for index in range(group, 50, 5)],
            )
            for group in range(5)
        ],
        latency_calls=[
            RerankMeasurement(call=call.model_copy(deep=True), scores=[0.5] * 20)
            for _ in range(50)
        ],
    )


def test_identical_order_has_perfect_correlation_and_ties_use_average_ranks() -> None:
    assert ranks([5, 2, 2]) == [2, 0.5, 0.5]
    result = compare(artifact("fp16"), artifact("fp32"))
    assert result.passed
    assert result.spearman == pytest.approx(1)
    assert artifact("fp16").scores == [index / 50 for index in range(50)]


def test_incompatible_or_constant_evidence_cannot_pass() -> None:
    other = artifact("fp32")
    other.input_sha256 = "b" * 64
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), other)
    other = artifact("fp32")
    for item in other.precision_calls:
        item.scores = [0.5] * 10
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), other)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha", "c" * 40),
        ("image_id", "sha256:" + "c" * 64),
        ("gpu_uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ],
)
def test_comparison_requires_same_observed_deployment(field: str, value: str) -> None:
    other = artifact("fp32")
    setattr(other.provenance, field, value)
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), other)


@pytest.mark.parametrize("stage", ["embedding", "relevance", "precision", "latency"])
def test_recovered_or_changed_response_settings_cannot_pass(stage: str) -> None:
    result = artifact("fp16")
    calls = {
        "embedding": result.embedding,
        "relevance": result.relevance.call,
        "precision": result.precision_calls[0].call,
        "latency": result.latency_calls[-1].call,
    }
    calls[stage].metadata.rerank_batch = 8
    assert not result.accepted
    assert not result.stable_settings
    with pytest.raises(ValidationError):
        compare(result, artifact("fp32"))


def test_comparison_rejects_changed_model_revision_even_if_each_run_is_stable() -> None:
    other = artifact("fp32")
    payload = other.model_dump_json().replace(other.metadata.embed_revision, "c" * 40)
    with pytest.raises(ValidationError):
        compare(artifact("fp16"), Benchmark.model_validate_json(payload))


def test_failed_quality_and_latency_cannot_be_hidden_by_matching_ranks() -> None:
    left, right = artifact("fp16"), artifact("fp32")
    for item in left.latency_calls:
        item.call.client_seconds = 2.01
    assert not compare(left, right).passed
    left = artifact("fp16")
    right.relevance.scores = [0.1, 0.9]
    assert not compare(left, right).passed
    right = artifact("fp32")
    for item in right.precision_calls:
        item.scores = [1 - score for score in item.scores]
    assert not compare(left, right).passed


def test_percentiles_and_default_configuration_are_computed_from_raw_calls() -> None:
    result = artifact("fp16")
    for index, item in enumerate(result.latency_calls):
        item.call.client_seconds = index / 25
    assert result.p50_s == pytest.approx(0.98)
    assert result.p95_s == pytest.approx(1.88)
    assert result.accepted
    payload = result.model_dump_json().replace('"rerank_batch":16', '"rerank_batch":8')
    assert not Benchmark.model_validate_json(payload).accepted


@pytest.mark.parametrize("damage", ["v1", "missing", "scores", "nan", "negative", "claimed"])
def test_incomplete_or_untrusted_evidence_is_rejected(damage: str) -> None:
    payload = json.loads(artifact("fp16").model_dump_json())
    if damage == "v1":
        payload["schema_version"] = 1
    elif damage == "missing":
        payload["latency_calls"].pop()
    elif damage == "scores":
        payload["precision_calls"][0]["scores"].pop()
    elif damage == "nan":
        payload["embedding"]["client_seconds"] = float("nan")
    elif damage == "negative":
        payload["embedding"]["queue_ms"] = -1
    else:
        payload["accepted"] = True
    with pytest.raises(SchemaValidationError):
        Benchmark.model_validate_json(json.dumps(payload))


def test_raw_artifacts_round_trip_without_trusting_a_saved_acceptance_flag() -> None:
    result = artifact("fp16")
    loaded = Benchmark.model_validate_json(result.model_dump_json())
    assert loaded == result
    assert loaded.accepted
