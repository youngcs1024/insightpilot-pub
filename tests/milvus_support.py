"""Real production Compose storage services; no model, GPU or private credentials."""

import json
import re
import secrets
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from scripts.ci_process import CommandRecorder, retain_primary_failure
from scripts.ci_storage import ContainerSample, StackEvidence, StorageIdentity, StorageSample
from app.retrieval.config import MilvusSettings
from scripts.deployment import DeploymentError
from tests.storage_lifecycle import CollectionOwner
from tests.storage_tracking import FAILED
from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    DeploymentSettings,
    MinioSettings,
    process_environment,
)
from tests.shared_database import require_docker

EVIDENCE = ROOT / "milvus-evidence"


def memory_bytes(value: str) -> int:
    """Parse Docker's documented unit-bearing memory counter without prose matching."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB|TiB)", value)
    if match is None:
        raise DeploymentError("Invalid Docker memory counter")
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    return int(float(match[1]) * units[match[2]])


class MilvusStack(BaseModel):
    """Only the test fixture publishes a loopback port and controls service restarts."""

    uri: str
    command: list[str]
    recorder: CommandRecorder = Field(repr=False)
    evidence: StackEvidence
    directory: Path

    @property
    def owner(self) -> CollectionOwner:
        """Keep ownership records in this stack only."""
        return CollectionOwner(self.uri, self.evidence, self.directory)

    @asynccontextmanager
    async def collection(
        self, prefix: str, request: pytest.FixtureRequest
    ) -> AsyncIterator[MilvusSettings]:
        """Collect failure-time resource observations before releasing the test collection."""
        async with self.owner.collection(
            prefix, request.node.nodeid, lambda: request.node.stash.get(FAILED, False)
        ) as settings:
            primary: BaseException | None = None
            try:
                yield settings
            except BaseException as exc:
                primary = exc
                raise
            finally:
                if primary is not None or request.node.stash.get(FAILED, False):
                    try:
                        await asyncio.to_thread(self.sample, "failure")
                    except Exception:
                        if primary is None:
                            raise
                        primary.add_note("Failure-time resource sample unavailable; see command evidence.")

    def sample(self, phase: str) -> None:
        """Read only container IDs resolved by this exact Compose project."""
        containers = []
        for service in ("milvus", "etcd", "minio"):
            identifier = self.recorder.run(
                "resource-id", [*self.command, "ps", "--all", "--quiet", service]
            ).stdout.strip()
            if not identifier or len(identifier.splitlines()) != 1:
                raise DeploymentError("Missing fixture-owned container")
            template = (
                '{"project":{{json (index .Config.Labels "com.docker.compose.project")}},'
                '"running":{{.State.Running}},"oom_killed":{{.State.OOMKilled}},'
                '"restarts":{{.RestartCount}},"limit_bytes":{{.HostConfig.Memory}}}'
            )
            info = json.loads(self.recorder.run(
                "resource-state", [self.command[0], "inspect", "--format", template, identifier]
            ).stdout)
            if info.pop("project") != self.evidence.project:
                raise DeploymentError("Container belongs to another project")
            usage = self.recorder.run(
                "resource-memory",
                [self.command[0], "stats", "--no-stream", "--format", "{{.MemUsage}}", identifier],
            ).stdout.split("/", 1)[0].strip()
            containers.append(ContainerSample(service=service, memory_bytes=memory_bytes(usage), **info))
        self.evidence.samples.append(StorageSample(phase=phase, containers=containers))
        self.owner.save()

    def finish(self) -> None:
        """Persist and enforce cleanup completion before stopping the isolated stack."""
        self.sample("final")
        self.evidence.completed = True
        self.owner.save()
        if not self.evidence.accepted:
            raise DeploymentError("Incomplete storage lifecycle evidence")

    def restart(self) -> None:
        """Restart only this fixture's Milvus, retaining its etcd and object-store state."""
        self.recorder.run("restart", [*self.command, "restart", "milvus"], timeout=60)
        self.recorder.run(
            "restart-wait",
            [*self.command, "up", "-d", "--wait", "--wait-timeout", "180", "milvus"],
            timeout=210,
        )


def record_storage_users(stack: MilvusStack) -> None:
    """Check declared and effective numeric users through the Docker daemon."""
    identifiers = stack.recorder.run(
        "container-ids",
        [*stack.command, "ps", "--quiet", "etcd", "minio", "milvus"],
    ).stdout.splitlines()
    expected_services = 3
    assert len(identifiers) == expected_services
    users: dict[str, list[str]] = {}
    for identifier in identifiers:
        configured = stack.recorder.run(
            "configured-user",
            [stack.command[0], "inspect", "--format", "{{.Config.User}}", identifier],
        ).stdout.strip()
        assert configured == "10001:10001"
        processes = stack.recorder.run(
            "effective-users",
            [stack.command[0], "top", identifier, "-eo", "pid,uid"],
        ).stdout.splitlines()[1:]
        assert processes
        rows = [line.split() for line in processes]
        expected_columns = 2
        assert all(len(row) == expected_columns for row in rows)
        assert all(row[0].isdigit() and row[1].isdigit() for row in rows)
        assert all(int(row[1]) != 0 for row in rows)
        users[identifier] = [row[1] for row in rows]
    (stack.directory / "runtime-users.json").write_text(json.dumps(users, indent=2))


@pytest.fixture(scope="session")
def milvus_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MilvusStack]:
    """Share service startup, but every test owns a fresh collection and async client."""
    docker = require_docker()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name="insightpilot-test-milvus-" + uuid4().hex[:12],
        postgres_superuser_password=secrets.token_urlsafe(24),
        bootstrap=BootstrapSecrets(
            app_password=secrets.token_urlsafe(24),
            etl_password=secrets.token_urlsafe(24),
            mcp_password=secrets.token_urlsafe(24),
        ),
        minio=MinioSettings(user="step31test", password=secrets.token_urlsafe(24)),
    )
    overlay = tmp_path_factory.mktemp("milvus") / "ports.yml"
    # Docker 29 cannot publish an internal-only endpoint; production stays internal.
    overlay.write_text(
        f"services:\n  milvus:\n    ports: ['127.0.0.1:{port}:19530']\n"
        "    networks: [backend, egress]\n"
    )
    command = [
        docker,
        "compose",
        "-p",
        settings.compose_project_name,
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(ROOT / ".env.deployment.example"),
        "-f",
        str(ROOT / "docker-compose.yml"),
        "-f",
        str(overlay),
        "--profile",
        "retrieval",
    ]
    environment = process_environment(settings)
    identity = StorageIdentity()
    directory = EVIDENCE / identity.tested_sha / (identity.run_id + "-" + identity.run_attempt) / settings.compose_project_name
    directory.mkdir(parents=True, exist_ok=True)
    recorder = CommandRecorder(
        directory=directory / "commands",
        cwd=ROOT,
        environment=environment,
        secrets=[
            settings.postgres_superuser_password,
            settings.bootstrap.app_password,
            settings.bootstrap.etl_password,
            settings.bootstrap.mcp_password,
            *([settings.minio.user] if settings.minio.user else []),
            *([settings.minio.password] if settings.minio.password else []),
        ],
    )
    stack = MilvusStack(
        uri=f"http://127.0.0.1:{port}", command=command, recorder=recorder, directory=directory,
        evidence=StackEvidence(identity=identity, project=settings.compose_project_name),
    )
    stack.owner.save()
    (directory / "stack.json").write_text(json.dumps({"project": settings.compose_project_name}))
    with retain_primary_failure(
        [
            stack.finish,
            lambda: recorder.run("final-state", [*command, "ps", "--all", "--format", "json"]),
            lambda: recorder.run(
                "service-logs", [*command, "logs", "--no-color", "etcd", "minio", "milvus"]
            ),
            lambda: recorder.run("stop", [*command, "stop", "milvus", "minio", "etcd"], timeout=90),
        ]
    ):
        recorder.run("build", [*command, "build", "etcd", "minio", "milvus"], timeout=240)
        recorder.run(
            "startup",
            [*command, "up", "-d", "--wait", "--wait-timeout", "180", "etcd", "minio", "milvus"],
            timeout=240,
        )
        record_storage_users(stack)
        stack.sample("startup")
        yield stack
