"""Exercise measurement capture with typed HTTP substitutes, never a GPU."""
# ruff: noqa: PLR2004 -- exact dedicated-workload dimensions.

import httpx
import pytest
import respx
from tenacity import wait_none

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.schemas.model_runtime import EmbedOutput, EmbedResult, ReadyResult, RerankResult
from model_runtime.errors import ModelContractError, ModelError
from scripts import bench_model_runtime
from tests.fakes.model_evidence import artifact
from tests.model_client_support import (
    BASE_URL,
    embed_output,
    embedding_response,
    failure,
    settings as model_settings,
)


@pytest.mark.parametrize("recover", [False, True])
async def test_benchmark_records_every_effective_response(
    monkeypatch: pytest.MonkeyPatch, recover: bool
) -> None:
    expected = artifact("fp16")
    reranks = []
    closed = []

    async def ready(self: ModelRuntimeClient, **kwargs: object) -> ReadyResult:
        return ReadyResult(request_id="ready", metadata=expected.metadata)

    async def embed(self: ModelRuntimeClient, *args: object, **kwargs: object) -> EmbedOutput:
        response = EmbedResult(
            request_id="embed",
            metadata=expected.metadata,
            ms=10,
            queue_ms=2,
            inference_ms=8,
            dense=[[0.0] * 1024 for _ in range(16)],
            sparse=[{42: 0.5} for _ in range(16)],
        )
        return embed_output(response)

    async def rerank(
        self: ModelRuntimeClient, query: str, passages: list[str], **kwargs: object
    ) -> RerankResult:
        metadata = expected.metadata.model_copy(deep=True)
        if recover and len(reranks) == 6:
            metadata.rerank_batch = 8
        reranks.append(len(passages))
        return RerankResult(
            request_id=f"rerank-{len(reranks)}",
            metadata=metadata,
            scores=[1 / (index + 1) for index in range(len(passages))],
            ms=20,
            queue_ms=3,
            inference_ms=17,
        )

    async def close(self: ModelRuntimeClient) -> None:
        closed.append(True)

    # Avoid process settings and any transport while retaining the real client boundary.
    client = object.__new__(ModelRuntimeClient)
    monkeypatch.setattr(bench_model_runtime, "ModelRuntimeClient", lambda settings: client)
    settings = bench_model_runtime.ModelDiagnosticsSettings(
        model_runtime=ModelRuntimeClientSettings(auth_token="synthetic"),  # noqa: S106 -- public fixture credential.
        _env_file=None,
    )
    monkeypatch.setattr(bench_model_runtime.ModelDiagnosticsSettings, "load", lambda: settings)
    monkeypatch.setattr(ModelRuntimeClient, "ready", ready)
    monkeypatch.setattr(ModelRuntimeClient, "embed", embed)
    monkeypatch.setattr(ModelRuntimeClient, "rerank", rerank)
    monkeypatch.setattr(ModelRuntimeClient, "aclose", close)
    result = await bench_model_runtime.benchmark(expected.provenance)
    assert reranks == [2, *([10] * 5), *([20] * 50)]
    assert result.embedding.request_id == "embed"
    assert result.latency_calls[0].call.queue_ms == 3
    assert result.latency_calls[0].call.inference_ms == 17
    assert result.latency_calls[0].call.metadata.rerank_batch == (8 if recover else 16)
    assert result.accepted is not recover
    assert closed == [True]


def test_shared_artifact_factory_returns_independent_objects() -> None:
    left, right = artifact("fp16"), artifact("fp16")
    left.latency_calls[0].call.metadata.rerank_batch = 8
    assert right.accepted
    assert right.latency_calls[0].call.metadata.rerank_batch == 16


@pytest.mark.parametrize("case", ["split", "retried"])
async def test_single_request_benchmark_rejects_multiple_embedding_requests(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, case: str,
) -> None:
    config = bench_model_runtime.ModelDiagnosticsSettings(
        model_runtime=model_settings(embed_batch=8 if case == "split" else 16), _env_file=None,
    )
    monkeypatch.setattr(bench_model_runtime.ModelDiagnosticsSettings, "load", lambda: config)
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())
    ready = ReadyResult(request_id="ready", metadata=artifact("fp16").metadata)
    respx_mock.get(BASE_URL + "/ready").mock(return_value=httpx.Response(200, json=ready.model_dump(mode="json")))
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if case == "retried" and calls == 1:
            return failure(ModelError())
        return embedding_response(request)

    respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    with pytest.raises(ModelContractError):
        await bench_model_runtime.benchmark(artifact("fp16").provenance)
    assert calls == 2
