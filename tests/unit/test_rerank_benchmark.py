"""Exercise dedicated benchmark data capture without real GPU inference."""
# ruff: noqa: PLR2004 -- fixed dedicated-workload dimensions and thresholds.

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.model_runtime import ReadyResult
from scripts import bench_retrieval_rerank as bench
from tests.fakes.model_evidence import artifact
from tests.rerank_support import model


@pytest.mark.parametrize("changed_metadata", [False, True])
async def test_benchmark_uses_shared_stage_and_preserves_identity(
    monkeypatch: pytest.MonkeyPatch, changed_metadata: bool
) -> None:
    reference = artifact("fp16")
    client = model([0.8] * 20)
    if changed_metadata:
        client.rerank.return_value.metadata.rerank_batch = 8
    client.ready = AsyncMock(
        return_value=ReadyResult(request_id="ready", metadata=reference.metadata)
    )
    client.aclose = AsyncMock()
    monkeypatch.setattr(
        bench.ModelDiagnosticsSettings, "load", lambda: SimpleNamespace(model_runtime=None)
    )
    monkeypatch.setattr(bench, "ModelRuntimeClient", lambda config: client)
    result = await bench.benchmark("a" * 40, reference.provenance)
    assert result.accepted is not changed_metadata
    assert client.rerank.await_count == 50
    assert all(len(call.args[1]) == 20 for call in client.rerank.call_args_list)
    assert all(call.kwargs["max_length"] == 320 for call in client.rerank.call_args_list)
    assert all(not item.result.candidates for item in result.measurements)
    assert "父章节" not in result.model_dump_json()
    assert result.p95_s is not None
    assert result.p50_s is not None
    assert all(item.result.response.queue_ms == 0 for item in result.measurements)
    client.aclose.assert_awaited_once()
    result.measurements.pop()
    assert not result.accepted
    assert result.p95_s is None


def test_workload_is_stable_and_has_distinct_child_parent_text() -> None:
    first, second = bench.workload(), bench.workload()
    assert first == second
    assert len(first) == 20
    assert all(item.content != item.parent_content for item in first)
    assert len({item.chunk_uuid for item in first}) == 20
