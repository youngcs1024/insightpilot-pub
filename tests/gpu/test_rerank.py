"""Explicit target-machine rerank latency acceptance; excluded before ordinary collection."""

from pathlib import Path

import pytest

from scripts.bench_retrieval_rerank import RerankBenchmark

pytestmark = [pytest.mark.gpu, pytest.mark.external]


def test_rerank_latency_under_budget(request: pytest.FixtureRequest) -> None:
    # The operator supplies a newly generated, SHA-bound benchmark artifact.
    # Loading the measurement is intentional: duplicate live inference is not needed.
    artifact = request.config.getoption("--rerank-evidence")
    assert artifact, "A current dedicated benchmark artifact is required"
    result = RerankBenchmark.model_validate_json(Path(artifact).read_text())
    assert result.client_sha == request.config.getoption("--client-sha")
    assert result.accepted
