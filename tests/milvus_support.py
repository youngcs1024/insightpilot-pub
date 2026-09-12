"""Real production Compose storage services; no model, GPU or private credentials."""

import json
import secrets
import socket
from collections.abc import Iterator
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from scripts.ci_process import CommandRecorder, retain_primary_failure
from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    DeploymentSettings,
    MinioSettings,
    process_environment,
)
from tests.shared_database import require_docker

EVIDENCE = ROOT / "milvus-evidence"


class MilvusStack(BaseModel):
    """Only the test fixture publishes a loopback port and controls service restarts."""

    uri: str
    command: list[str]
    recorder: CommandRecorder = Field(repr=False)

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
    (EVIDENCE / "runtime-users.json").write_text(json.dumps(users, indent=2))


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
    EVIDENCE.mkdir(exist_ok=True)
    recorder = CommandRecorder(
        directory=EVIDENCE / "commands",
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
    stack = MilvusStack(uri=f"http://127.0.0.1:{port}", command=command, recorder=recorder)
    (EVIDENCE / "stack.json").write_text(json.dumps({"project": settings.compose_project_name}))
    with retain_primary_failure(
        [
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
        yield stack
