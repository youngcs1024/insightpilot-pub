"""Capacity/security regressions run without company SSH, model artifacts or a GPU."""

# ruff: noqa: PLR2004 -- explicit protocol/status/fixture expectations.

import socket
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from spikes.capacity import observe, operator
from spikes.capacity.cgroup import memory_peak
from spikes.capacity.client import ProbeClient
from spikes.capacity.contracts import (
    GIB,
    PROJECT_CEILING,
    FailureKind,
    PairBatch,
    ProbeError,
    ResourceIdentity,
    Status,
    TextBatch,
    assess_capacity,
    percentile,
    spearman,
    verify_unchanged,
)
from spikes.capacity.corpus import corpus_hash, fixed_pairs
from spikes.capacity.model_probe import Models, create_app, guard_forward
from spikes.capacity.report import assemble
from spikes.capacity.settings import CapacityFields, ProbeLimits
from spikes.capacity.workload import classify_failure


@pytest.mark.parametrize("relative", ["memory.peak", "memory/memory.max_usage_in_bytes"])
def test_kernel_peak_supports_both_cgroup_versions(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("123456")
    assert memory_peak(tmp_path) == 123456


@pytest.mark.parametrize("value", [None, "invalid", "0", "-1"])
def test_missing_or_invalid_kernel_peak_is_typed(tmp_path: Path, value: str | None) -> None:
    if value is not None:
        (tmp_path / "memory.peak").write_text(value)
    with pytest.raises(ProbeError) as failure:
        memory_peak(tmp_path)
    assert failure.value.kind is FailureKind.SAMPLING


def test_forward_oom_cannot_trigger_upstream_runtimeerror_retry() -> None:
    class FakeOOMError(RuntimeError):
        pass

    torch = Mock()
    torch.cuda.OutOfMemoryError = FakeOOMError
    model = Mock()
    model.forward.side_effect = FakeOOMError("out of memory")
    batches: list[int] = []
    original = model.forward
    guard_forward(model, torch, batches)
    with pytest.raises(ProbeError) as failure:
        model.forward(input_ids=Mock(shape=(16, 512)))
    assert failure.value.kind is FailureKind.OOM
    assert not isinstance(failure.value, RuntimeError)
    assert batches == [16]
    original.assert_called_once()


def test_capacity_uses_current_docker_memtotal(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock(side_effect=[str(8 * GIB), str(16 * GIB)])
    monkeypatch.setattr(observe, "command", reader)
    assert observe.current_memtotal() == 8 * GIB
    assert observe.current_memtotal() == 16 * GIB
    assert reader.call_args.args[0] == ["docker", "info", "--format", "{{.MemTotal}}"]


def test_capacity_requires_one_gib_headroom() -> None:
    assert assess_capacity(8 * GIB - 1, 6 * GIB, GIB).status is Status.FAILED
    assert assess_capacity(8 * GIB, 6 * GIB, GIB).status is Status.PASSED


def test_capacity_accepts_expanded_allocation() -> None:
    assert assess_capacity(8 * GIB, 8 * GIB, 0).status is Status.FAILED
    assert assess_capacity(10 * GIB, 8 * GIB, 0).status is Status.PASSED


def test_decimal_ceiling_independent_of_binary_headroom() -> None:
    assert assess_capacity(16 * GIB, PROJECT_CEILING, 0).status is Status.PASSED
    result = assess_capacity(16 * GIB, PROJECT_CEILING + 1, 0)
    assert result.status is Status.FAILED
    assert result.recommended_allocation_bytes == 14 * GIB


@pytest.mark.parametrize(("own", "other"), [(None, 0), (0, None), (None, None)])
def test_missing_measurements_never_pass(own: int | None, other: int | None) -> None:
    assert assess_capacity(16 * GIB, own, other).status is Status.PENDING


def test_port_conflict_reports_without_killing(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = Mock(return_value="LISTEN pid=1234 name=foreign-service")
    monkeypatch.setattr(operator, "command", commands)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ProbeError, match="foreign-service"):
            operator.verify_port(port)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    assert commands.call_args.args[0] == ["ss", "-ltnp", f"sport = :{port}"]


def test_foreign_resources_unchanged() -> None:
    before = [ResourceIdentity(kind="container", id="abc", name="pathfinder", state="exited")]
    verify_unchanged(before, before)
    with pytest.raises(ProbeError) as error:
        verify_unchanged(before, [before[0].model_copy(update={"state": "running"})])
    assert error.value.kind is FailureKind.ISOLATION
    with pytest.raises(ProbeError):
        verify_unchanged(before, [])


def test_headroom_stop_is_scoped_to_exact_project(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock()
    reader.containers.return_value = [
        observe.DockerContainer(
            Id="own",
            Names=["probe"],
            Labels={"com.docker.compose.project": "insightpilot-model-test-step08-a"},
            State="running",
        ),
        observe.DockerContainer(
            Id="foreign",
            Names=["other"],
            Labels={"com.docker.compose.project": "pathfinder"},
            State="running",
        ),
    ]
    commands = Mock(return_value="")
    monkeypatch.setattr(observe, "command", commands)
    observe.stop_own_probe(reader, "insightpilot-model-test-step08-a")
    commands.assert_called_once_with(["docker", "stop", "--time", "2", "own"])


@pytest.mark.parametrize(
    "project", ["pathfinder", "insightpilot", "insightpilot-model", "--project-directory"]
)
def test_probe_rejects_nonisolated_project(project: str, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        CapacityFields(project=project, output=tmp_path / "report.json")


def test_fixed_pairs_are_varied_and_reproducible() -> None:
    assert len(fixed_pairs()) == 50
    assert len({p.model_dump_json() for p in fixed_pairs()}) > 20
    assert corpus_hash(1024) == corpus_hash(1024)
    assert corpus_hash(1024) != corpus_hash(1023)


def test_spearman_handles_ties_and_rank_reversal() -> None:
    assert spearman([1, 1, 3], [2, 2, 4]) == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


@pytest.mark.parametrize(
    ("left", "right"),
    [([], []), ([1], [1]), ([1, 1], [2, 3]), ([1, 2], [1]), ([1, float("nan")], [1, 2])],
)
def test_degenerate_comparisons_fail(left: list[float], right: list[float]) -> None:
    with pytest.raises(ProbeError):
        spearman(left, right)


def test_percentiles_use_nearest_rank() -> None:
    assert percentile(list(map(float, range(1, 101))), 0.95) == 95
    with pytest.raises(ProbeError):
        percentile([], 0.5)


@pytest.mark.parametrize("count", [0, 17])
def test_embedding_bounds(count: int) -> None:
    with pytest.raises(ValidationError):
        TextBatch(texts=["text"] * count)


async def test_probe_authentication_before_inference() -> None:
    models = Mock(spec=Models)
    app = create_app(models, "test-only-token")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://probe"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.post("/v1/embed", json={"texts": ["text"]})
        assert response.status_code == 401
        assert response.json() == {"kind": "authentication"}
    models.embed.assert_not_called()


@pytest.mark.parametrize("kind", [FailureKind.CUDA, FailureKind.OOM, FailureKind.HEADROOM])
async def test_typed_cuda_failure_over_http(kind: FailureKind) -> None:
    models = Mock(spec=Models)
    models.embed.side_effect = ProbeError(kind, "private diagnostic text")
    app = create_app(models, "test-only-token")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://probe"
    ) as client:
        response = await client.post(
            "/v1/embed",
            json={"texts": ["text"]},
            headers={"Authorization": "Bearer test-only-token"},
        )
    assert response.status_code == 503
    assert response.json() == {"kind": kind.value}
    assert "private" not in response.text


@pytest.mark.parametrize("failure", ["timeout", "transport", "invalid", "cuda", "auth"])
async def test_client_failure_is_typed_and_not_retried(failure: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("hidden", request=request)
        if failure == "transport":
            raise httpx.ConnectError("hidden", request=request)
        if failure == "cuda":
            return httpx.Response(503, json={"kind": "cuda"})
        if failure == "auth":
            return httpx.Response(401, json={"kind": "authentication"})
        return httpx.Response(200, content=b'{"dense":[[Infinity]]}')

    client = ProbeClient("http://probe", "test-token")
    await client.http.aclose()
    handler = Mock(side_effect=respond)
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://probe")
    try:
        with pytest.raises(ProbeError):
            await client.rerank(PairBatch(pairs=fixed_pairs()[:20]))
    finally:
        await client.http.aclose()
    handler.assert_called_once()


def test_headroom_exception_does_not_change_default() -> None:
    assert ProbeLimits().gpu_headroom_bytes == 4 * GIB
    assert ProbeLimits(gpu_headroom_bytes=0).gpu_headroom_bytes == 0
    with pytest.raises(ValidationError):
        ProbeLimits(gpu_headroom_bytes=-1)


def test_task_group_keeps_underlying_failure_category() -> None:
    error = ExceptionGroup("not inspected", [ProbeError(FailureKind.OOM, "not inspected")])
    assert classify_failure(error) is FailureKind.OOM
    assert classify_failure(TimeoutError()) is FailureKind.TIMEOUT


def test_sampling_failure_preserves_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Mock()
    reader.inventory.return_value = []
    reader.sample.side_effect = ProbeError(FailureKind.TIMEOUT, "bounded timeout")
    monkeypatch.setattr(observe, "DockerReader", Mock(return_value=reader))
    monkeypatch.setattr(observe, "current_memtotal", Mock(return_value=8 * GIB))
    monkeypatch.setattr(observe.time, "monotonic", Mock(side_effect=range(100)))
    output = tmp_path / "failed.json"
    result = observe.observe(
        CapacityFields(project="insightpilot-test-step08-test", output=output, duration_s=3)
    )
    assert result.status is Status.FAILED
    assert result.failure is FailureKind.TIMEOUT
    assert observe.Observation.model_validate_json(output.read_text()).status is Status.FAILED
    assert '"pending"' in observe.local_verdict(result)


def test_stop_signal_finishes_sampling_and_records_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Mock()
    reader.inventory.return_value = []
    reader.sample.return_value = []
    monkeypatch.setattr(observe, "DockerReader", Mock(return_value=reader))
    monkeypatch.setattr(observe, "current_memtotal", Mock(return_value=8 * GIB))
    monkeypatch.setattr(observe, "collect_peaks", Mock(return_value={"own": 123}))
    stop = tmp_path / "stop"
    stop.touch()
    result = observe.observe(
        CapacityFields(
            project="insightpilot-test-step08-test",
            output=tmp_path / "observation.json",
            stop_file=stop,
        )
    )
    assert result.status is Status.PASSED
    reader.sample.assert_called_once()
    assert len(result.disk_samples) == 2
    assert all(sample.free_bytes > 0 for sample in result.disk_samples)


def test_partial_evidence_never_passes_step_gate(tmp_path: Path) -> None:
    result = assemble(tmp_path)
    assert result.status is Status.PENDING
    assert len(result.pending) == 6
    assert result.full_application_capacity is Status.PENDING


def test_no_gpu_import_on_ordinary_collection() -> None:
    assert "torch" not in sys.modules
    assert "FlagEmbedding" not in sys.modules


async def test_invalid_upstream_output_is_typed() -> None:
    models = Mock(spec=Models)
    with pytest.raises(ValidationError) as error:
        TextBatch(texts=[])
    models.embed.side_effect = error.value
    app = create_app(models, "test-token")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://probe"
    ) as client:
        response = await client.post(
            "/v1/embed", json={"texts": ["text"]}, headers={"Authorization": "Bearer test-token"}
        )
    assert response.status_code == 503
    assert response.json() == {"kind": "output"}


def test_observer_rejects_missing_memory_counters() -> None:
    with pytest.raises(ValidationError):
        observe.DockerStats.model_validate({"memory_stats": {}})


def test_gpu_identity_mismatch_is_sampling_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observe, "command", Mock(return_value="GPU-other, 24000, 1, 23999"))
    with pytest.raises(ProbeError) as error:
        observe.gpu_reading("GPU-expected", 0)
    assert error.value.kind is FailureKind.SAMPLING


def test_kernel_reader_is_readonly_and_project_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Mock()
    project = "insightpilot-test-step08-test"
    reader.containers.return_value = [
        observe.DockerContainer(
            Id="abc",
            Names=["etcd"],
            Labels={"com.docker.compose.project": project},
            State="running",
        ),
        observe.DockerContainer(Id="foreign", Names=["foreign"], Labels={}, State="running"),
    ]
    commands = Mock(return_value='{"abc": 123}')
    monkeypatch.setattr(observe, "command", commands)
    assert observe.collect_peaks(
        reader, CapacityFields(project=project, output=tmp_path / "result")
    ) == {"abc": 123}
    args = commands.call_args.args[0]
    assert "--read-only" in args
    assert "type=bind,src=/sys/fs/cgroup,dst=/host-cgroup,readonly" in args
    assert "foreign" not in args
    assert "--privileged" not in args
    assert "--rm" not in args


def test_missing_kernel_counter_fails_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Mock()
    project = "insightpilot-test-step08-test"
    reader.containers.return_value = [
        observe.DockerContainer(
            Id="abc",
            Names=["etcd"],
            Labels={"com.docker.compose.project": project},
            State="running",
        )
    ]
    monkeypatch.setattr(observe, "command", Mock(return_value="{}"))
    with pytest.raises(ProbeError):
        observe.collect_peaks(reader, CapacityFields(project=project, output=tmp_path / "result"))
