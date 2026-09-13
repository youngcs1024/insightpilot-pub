"""Synthetic versioned model evidence shared by ordinary tests."""

from scripts.model_evidence import Benchmark, CallMeasurement, Provenance, RerankMeasurement
from tests.fakes.model_runtime import FakeModels


def artifact(precision: str) -> Benchmark:
    """Return an independent complete synthetic benchmark."""
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
            RerankMeasurement(call=call.model_copy(deep=True), scores=[0.5] * 20) for _ in range(50)
        ],
    )
