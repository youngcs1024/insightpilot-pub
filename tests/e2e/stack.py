"""Fixture-owned Compose lifecycle; imports never contact Docker or a service."""

import secrets
import socket
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel, SecretStr

from scripts.ci_process import CommandRecorder, retain_primary_failure
from scripts.ci_storage import StorageIdentity
from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    DeploymentSettings,
    MinioSettings,
    process_environment,
)
from tests.database_support import foreign_snapshot
from tests.shared_database import require_docker


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class E2EStack(BaseModel):
    api_url: str
    inference_url: str
    token: SecretStr
    command: list[str]
    recorder: CommandRecorder
    directory: Path

    def api_container_id(self, stage: str) -> str:
        """Read the running API identity to prove an MCP restart kept it alive."""
        result = self.recorder.run(stage, [*self.command, "ps", "-q", "api"])
        assert result.stdout.strip()
        return result.stdout.strip()

    def restart_api(self) -> None:
        self.recorder.run("restart-api", [*self.command, "restart", "api"], timeout=60)
        self.recorder.run(
            "wait-api",
            [*self.command, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "90", "api"],
            timeout=120,
        )

    def stop(self, service: str) -> None:
        assert service in {"mcp", "inference"}
        self.recorder.run(
            "stop-" + service, [*self.command, "stop", "-t", "1", service], timeout=30
        )

    def start_mcp(self) -> None:
        """Restore only MCP; the API must recover its existing client session."""
        self.recorder.run(
            "start-mcp",
            [*self.command, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "90", "mcp"],
            timeout=120,
        )

    def restore(self) -> None:
        self.recorder.run(
            "restore",
            [
                *self.command,
                "up",
                "-d",
                "--no-deps",
                "--wait",
                "--wait-timeout",
                "90",
                "mcp",
                "inference",
            ],
            timeout=120,
        )
        self.restart_api()


@pytest.fixture(scope="session")
def e2e_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[E2EStack]:
    docker = require_docker()
    before = foreign_snapshot(docker)
    identifier = "insightpilot-test-e2e-" + uuid4().hex[:12]
    api_port, inference_port = free_port(), free_port()
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name=identifier,
        api_host_port=api_port,
        postgres_superuser_password=secrets.token_urlsafe(24),
        bootstrap=BootstrapSecrets(
            **{
                name + "_password": secrets.token_urlsafe(24)
                for name in ("app", "etl", "mcp", "audit")
            }
        ),
        mcp_auth_token=secrets.token_urlsafe(24),
        jwt_secret=secrets.token_urlsafe(48),
        minio=MinioSettings(user="e2e-user", password=secrets.token_urlsafe(24)),
    )
    identity = StorageIdentity()
    directory = (
        ROOT
        / "e2e-evidence"
        / identity.tested_sha
        / (identity.run_id + "-" + identity.run_attempt)
        / identifier
    )
    directory.mkdir(parents=True)
    (directory / "identity.json").write_text(identity.model_dump_json())
    empty_env = tmp_path_factory.mktemp("e2e-compose") / "empty.env"
    empty_env.write_text("")
    environment = process_environment(settings)
    environment.update(
        {
            "IP_E2E_API_BASE": identifier + ":base",
            "IP_E2E_API_IMAGE": identifier + ":test",
            "IP_E2E_MCP_IMAGE": identifier + ":mcp",
            "IP_E2E_INFERENCE_PORT": str(inference_port),
        }
    )
    recorder = CommandRecorder(
        directory=directory / "commands",
        cwd=ROOT,
        environment=environment,
        secrets=[
            settings.postgres_superuser_password,
            settings.bootstrap.app_password,
            settings.bootstrap.etl_password,
            settings.bootstrap.mcp_password,
            settings.mcp_auth_token,
            settings.jwt_secret,
            settings.minio.password,
        ],
    )
    command = [
        docker,
        "compose",
        "-p",
        identifier,
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(empty_env),
        "-f",
        str(ROOT / "docker-compose.e2e.yml"),
    ]
    stack = E2EStack(
        api_url=f"http://127.0.0.1:{api_port}",
        inference_url=f"http://127.0.0.1:{inference_port}",
        token=settings.mcp_auth_token,
        command=command,
        recorder=recorder,
        directory=directory,
    )
    with retain_primary_failure(
        [
            lambda: recorder.run("final-state", [*command, "ps", "--all", "--format", "json"]),
            lambda: recorder.run("logs", [*command, "logs", "--no-color"]),
            lambda: recorder.run("stop", [*command, "stop"], timeout=90),
        ]
    ):
        recorder.run(
            "build-base",
            [docker, "build", "-f", "docker/Dockerfile.api", "-t", identifier + ":base", "."],
            timeout=600,
        )
        recorder.run(
            "build-services",
            [*command, "build", "api", "mcp", "etcd", "minio", "milvus"],
            timeout=600,
        )
        recorder.run(
            "startup", [*command, "up", "-d", "--wait", "--wait-timeout", "240", "api"], timeout=300
        )
        yield stack
    assert foreign_snapshot(docker) == before
