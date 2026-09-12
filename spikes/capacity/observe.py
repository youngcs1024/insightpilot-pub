"""Run separately on each host; never wraps remote commands or manages a foreign service."""

import json
import shutil
import subprocess
import time
from typing import Annotated
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from spikes.capacity.contracts import (
    FailureKind,
    MemorySample,
    ProbeError,
    ResourceIdentity,
    Status,
    assess_capacity,
    verify_unchanged,
)
from spikes.capacity.settings import CapacityFields, CapacitySettings


class DockerContainer(BaseModel):
    """Small read adapter for Docker's third-party container JSON."""

    id: str = Field(alias="Id")
    names: list[str] = Field(alias="Names")
    labels: dict[str, str] = Field(alias="Labels")
    state: str = Field(alias="State")


class DockerResource(BaseModel):
    """Network and volume list adapter."""

    id: str = Field(default="", alias="Id")
    name: str = Field(alias="Name")
    labels: dict[str, str] | None = Field(default=None, alias="Labels")


class MemoryStats(BaseModel):
    """Extract only byte-valued memory counters from Docker stats."""

    usage: int = Field(ge=0)
    limit: int = Field(gt=0)
    stats: dict[str, int]


class DockerStats(BaseModel):
    """A stopped container or malformed response must fail sampling."""

    memory_stats: MemoryStats


class GPUReading(BaseModel):
    """One authorized device's global usage includes co-tenants."""

    elapsed_s: float
    uuid: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


class DiskReading(BaseModel):
    """Container root and evidence filesystem availability, in bytes."""

    elapsed_s: float
    path: str
    total_bytes: int
    free_bytes: int


class Observation(BaseModel):
    """Immutable run result; failures retain partial observations."""

    status: Status = Status.PENDING
    failure: FailureKind | None = None
    failure_detail: str | None = None
    project: str
    memtotal_before_bytes: int
    memtotal_after_bytes: int = 0
    interval_s: float
    duration_s: float = 0
    samples: list[MemorySample] = Field(default_factory=list)
    gpu_samples: list[GPUReading] = Field(default_factory=list)
    disk_samples: list[DiskReading] = Field(default_factory=list)
    cgroup_peaks: dict[str, int] = Field(default_factory=dict)
    before: list[ResourceIdentity]
    after: list[ResourceIdentity] = Field(default_factory=list)


def command(args: list[str], timeout: float = 10) -> str:
    """Read-only operator subprocesses are bounded and never retried implicitly."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(FailureKind.TIMEOUT, "Observation command timed out.") from exc
    except OSError as exc:
        raise ProbeError(FailureKind.PREREQUISITE, "Observation executable unavailable.") from exc
    if result.returncode:
        raise ProbeError(FailureKind.SAMPLING, "Observation command returned a failure exit code.")
    return result.stdout.strip()


def current_memtotal() -> int:
    """Read the running engine each time, never a configured WSL number."""
    return TypeAdapter(Annotated[int, Field(gt=0)]).validate_python(
        command(["docker", "info", "--format", "{{.MemTotal}}"])
    )


class DockerReader:
    """Read Docker locally using a bounded Unix-socket HTTP connection."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds="/var/run/docker.sock"),
            base_url="http://docker/v1.47",
            timeout=5,
        )

    def read(self, path: str) -> bytes:
        """No automatic retry: an unobserved interval cannot certify a peak."""
        try:
            response = self.client.get(path)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProbeError(FailureKind.SAMPLING, "Docker sampling request failed.") from exc
        return response.content

    def containers(self) -> list[DockerContainer]:
        """Return all containers including stopped foreign projects."""
        return TypeAdapter(list[DockerContainer]).validate_json(self.read("/containers/json?all=1"))

    def inventory(self, project: str) -> list[ResourceIdentity]:
        """Exclude only this exact disposable project, never every InsightPilot resource."""
        resources = [
            ResourceIdentity(
                kind="container",
                id=c.id,
                name=c.names[0],
                state=c.state,
                project=c.labels.get("com.docker.compose.project", ""),
            )
            for c in self.containers()
        ]
        networks = TypeAdapter(list[DockerResource]).validate_json(self.read("/networks"))
        volumes = TypeAdapter(list[DockerResource]).validate_python(
            json.loads(self.read("/volumes"))["Volumes"] or []
        )
        for kind, items in (("network", networks), ("volume", volumes)):
            resources.extend(
                ResourceIdentity(
                    kind=kind,
                    id=item.id or item.name,
                    name=item.name,
                    project=(item.labels or {}).get("com.docker.compose.project", ""),
                )
                for item in items
            )
        return [r for r in resources if r.project != project]

    def sample(self, elapsed: float) -> list[MemorySample]:
        """Keep raw usage (conservative capacity gate) and working set separately."""
        samples = []
        for container in self.containers():
            if container.state != "running":
                continue
            try:
                stats = DockerStats.model_validate_json(
                    self.read(f"/containers/{container.id}/stats?stream=false&one-shot=true")
                ).memory_stats
            except ValidationError as exc:
                current = next((c for c in self.containers() if c.id == container.id), None)
                if current is not None and current.state != "running":
                    continue  # Normal one-shot exit; retain its earlier samples and job receipt.
                raise ProbeError(
                    FailureKind.SAMPLING, "A running container lost memory counters."
                ) from exc
            cache = stats.stats.get("inactive_file", stats.stats.get("total_inactive_file", 0))
            samples.append(
                MemorySample(
                    elapsed_s=elapsed,
                    container_id=container.id,
                    project=container.labels.get("com.docker.compose.project", ""),
                    usage_bytes=stats.usage,
                    working_set_bytes=max(0, stats.usage - cache),
                    limit_bytes=stats.limit,
                )
            )
        return samples


def gpu_reading(uuid: str, elapsed: float) -> GPUReading:
    """Query the specified UUID, not whichever GPU happens to be index zero."""
    row = command(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-gpu=uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ).split(",")
    if len(row) != 4 or row[0].strip() != uuid:  # noqa: PLR2004
        raise ProbeError(FailureKind.SAMPLING, "GPU identity/reading mismatch.")
    total, used, free = (int(value.strip()) * 1024**2 for value in row[1:])
    return GPUReading(
        elapsed_s=elapsed, uuid=uuid, total_bytes=total, used_bytes=used, free_bytes=free
    )


def stop_own_probe(reader: DockerReader, project: str) -> None:
    """Only exact-label matching containers may be stopped for GPU headroom."""
    for container in reader.containers():
        if (
            container.labels.get("com.docker.compose.project") == project
            and container.state == "running"
        ):
            command(["docker", "stop", "--time", "2", container.id])


def collect_peaks(reader: DockerReader, config: CapacityFields) -> dict[str, int]:
    """Read host cgroups read-only; distroless etcd has no cat executable."""
    ids = [
        c.id
        for c in reader.containers()
        if c.labels.get("com.docker.compose.project") == config.project and c.state == "running"
    ]
    if not ids:
        return {}
    script = (
        "import json,sys; from pathlib import Path; "
        "root=Path('/host-cgroup'); "
        "paths=[*root.rglob('memory.peak'),*root.rglob('memory.max_usage_in_bytes')]; "
        "print(json.dumps({i:int(p.read_text()) for i in sys.argv[1:] for p in paths "
        "if p.parent.name in (i,'docker-'+i+'.scope')}))"
    )
    output = command(
        [
            "docker",
            "run",
            "--name",
            f"{config.project}-reader-{uuid4().hex[:8]}",
            "--label",
            f"com.docker.compose.project={config.project}",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "64m",
            "--cpus",
            "0.2",
            "--cgroupns",
            "host",
            "--mount",
            "type=bind,src=/sys/fs/cgroup,dst=/host-cgroup,readonly",
            "--entrypoint",
            "python",
            config.reader_image,
            "-c",
            script,
            *ids,
        ],
        timeout=30,
    )
    peaks = TypeAdapter(dict[str, Annotated[int, Field(ge=0)]]).validate_json(output)
    if set(peaks) != set(ids):
        raise ProbeError(FailureKind.SAMPLING, "A live probe container has no kernel peak counter.")
    return peaks


def sample_gpu(
    reader: DockerReader, config: CapacityFields, result: Observation, elapsed: float
) -> None:
    """Enforce admission before accumulating further GPU samples."""
    if config.gpu_id is None:
        return
    gpu = gpu_reading(config.gpu_id, elapsed)
    result.gpu_samples.append(gpu)
    if gpu.free_bytes < config.gpu_headroom_bytes:
        stop_own_probe(reader, config.project)
        raise ProbeError(
            FailureKind.HEADROOM, "Shared GPU headroom fell below the configured reserve."
        )


def observe(config: CapacityFields) -> Observation:
    """Save evidence even on a failed interval or a shared-memory admission failure."""
    reader = DockerReader()
    result = Observation(
        project=config.project,
        memtotal_before_bytes=current_memtotal(),
        interval_s=config.interval_s,
        before=reader.inventory(config.project),
    )
    start = time.monotonic()
    try:
        while time.monotonic() - start < config.duration_s:
            elapsed = time.monotonic() - start
            result.samples.extend(reader.sample(elapsed))
            sample_gpu(reader, config, result, elapsed)
            for path in ("/", str(config.output.parent)):
                disk = shutil.disk_usage(path)
                result.disk_samples.append(
                    DiskReading(
                        elapsed_s=elapsed,
                        path=path,
                        total_bytes=disk.total,
                        free_bytes=disk.free,
                    )
                )
            if config.stop_file is not None and config.stop_file.is_file():
                break
            time.sleep(config.interval_s)
        result.cgroup_peaks = collect_peaks(reader, config)
        result.status = Status.PASSED
    except ProbeError as exc:
        result.status, result.failure = Status.FAILED, exc.kind
        result.failure_detail = str(exc)
    except (ValidationError, ValueError, OSError):
        result.status, result.failure = Status.FAILED, FailureKind.SAMPLING
    finally:
        result.duration_s = time.monotonic() - start
        try:
            result.memtotal_after_bytes = current_memtotal()
            result.after = reader.inventory(config.project)
            verify_unchanged(result.before, result.after)
        except (ProbeError, ValidationError, ValueError, OSError):
            result.status, result.failure = Status.FAILED, FailureKind.ISOLATION
        reader.client.close()
        config.output.write_text(result.model_dump_json(indent=2) + "\n")
    return result


def local_verdict(observation: Observation, completed_job_peak: int = 0) -> str:
    """A conservative sum of per-container peaks covers non-simultaneous spikes too."""
    own = dict(observation.cgroup_peaks)
    other: dict[str, int] = {}
    for sample in observation.samples:
        destination = own if sample.project == observation.project else other
        destination[sample.container_id] = max(
            destination.get(sample.container_id, 0), sample.usage_bytes
        )
    complete = observation.status is Status.PASSED and bool(own)
    return assess_capacity(
        observation.memtotal_after_bytes,
        sum(own.values()) + completed_job_peak if complete else None,
        sum(other.values()) if complete else None,
    ).model_dump_json(indent=2)


def main() -> int:
    """Invoke locally or independently on the server with a process-specific env file."""
    config = CapacitySettings.load().capacity
    if config.output.exists():
        raise ProbeError(FailureKind.PREREQUISITE, "Choose a fresh evidence output path.")
    result = observe(config)
    print(local_verdict(result) if config.gpu_id is None else result.status.value)
    return 0 if result.status is Status.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
