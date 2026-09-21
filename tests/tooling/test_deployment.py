"""Validate the rendered deployment and reject foreign configuration or destructive flags."""

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.core.config_models import (
    DataAgentSettings,
    ObservabilitySettings,
    RouterSettings,
    SanitySettings,
)
from scripts.deployment import (
    ROOT,
    Command,
    DeploymentError,
    DeploymentSettings,
    Invocation,
    compose_prefix,
    execute,
    process_environment,
    run_command,
    validate_arguments,
    verify_port,
    verify_volume,
)


class Port(BaseModel):
    """Relevant Compose port contract."""

    target: int
    published: str
    host_ip: str


class Service(BaseModel):
    """Only fields needed to assert deployment isolation."""

    environment: dict[str, str] = Field(default_factory=dict)
    ports: list[Port] = Field(default_factory=list)
    profiles: list[str]


class ComposeConfig(BaseModel):
    """Typed rendered Compose boundary."""

    name: str
    services: dict[str, Service]


@pytest.fixture
def deployment_settings() -> DeploymentSettings:
    return DeploymentSettings(_env_file=ROOT / ".env.deployment.example")


def render(
    settings: DeploymentSettings, *, dev: bool = False, profile: str = "core"
) -> ComposeConfig:
    docker = shutil.which("docker")
    assert docker is not None, "Compose is required for configuration validation"
    call = Invocation(
        dev=dev,
        profiles=[profile],
        deployment_file=ROOT / ".env.deployment.example",
        command=Command.CONFIG,
        arguments=["--format", "json"],
    )
    return ComposeConfig.model_validate_json(execute(docker, settings, call))


def test_compose_project_override_cannot_target_pathfinder(
    deployment_settings: DeploymentSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "pathfinder")
    monkeypatch.setenv("COMPOSE_FILE", "/foreign/compose.yaml")
    monkeypatch.setenv("COMPOSE_ENV_FILES", "/foreign/.env")
    assert render(deployment_settings).name == "insightpilot"
    monkeypatch.setenv("IP_COMPOSE_PROJECT_NAME", "pathfinder")
    with pytest.raises(ValidationError):
        DeploymentSettings(_env_file=ROOT / ".env.deployment.example")


@pytest.mark.parametrize("profile", ["core", "retrieval", "full", "dev", "ui"])
def test_base_does_not_publish_database(
    deployment_settings: DeploymentSettings, profile: str
) -> None:
    config = render(deployment_settings, profile=profile)
    expected = {"postgres", "migrate", "mcp", "api"}
    if profile != "core":
        expected.update({"etcd", "minio", "milvus", "model-tunnel"})
    assert set(config.services) == expected
    assert not config.services["postgres"].ports


def test_dev_database_uses_15432(deployment_settings: DeploymentSettings) -> None:
    ports = render(deployment_settings, dev=True).services["postgres"].ports
    assert ports == [Port(target=5432, published="15432", host_ip="127.0.0.1")]


def test_service_secret_allowlists(deployment_settings: DeploymentSettings) -> None:
    env = render(deployment_settings).services["postgres"].environment
    assert set(env) == {
        "POSTGRES_PASSWORD",
        "IP_BOOTSTRAP_APP_PASSWORD",
        "IP_BOOTSTRAP_ETL_PASSWORD",
        "IP_BOOTSTRAP_MCP_PASSWORD",
    }
    assert "IP_COMPOSE_PROJECT_NAME" not in env


def test_retrieval_secrets_only_reach_storage(deployment_settings: DeploymentSettings) -> None:
    config = render(deployment_settings, profile="retrieval")
    assert config.services["minio"].environment["MINIO_ROOT_PASSWORD"]
    assert config.services["milvus"].environment["MINIO_SECRET_ACCESS_KEY"]
    for name in ("api", "mcp", "postgres", "migrate", "etcd"):
        assert not any("MINIO" in key for key in config.services[name].environment)
    for name in ("etcd", "minio", "milvus"):
        assert not config.services[name].ports


def test_retrieval_settings_reach_api(deployment_settings: DeploymentSettings) -> None:
    deployment_settings.retrieval.milvus.search_ef = 128
    config = render(deployment_settings, profile="retrieval")
    value = json.loads(config.services["api"].environment["IP_RETRIEVAL"])
    assert value["milvus"]["search_ef"] == deployment_settings.retrieval.milvus.search_ef
    assert value["enabled"] is False


def test_tracing_credentials_only_reach_api(deployment_settings: DeploymentSettings) -> None:
    deployment_settings.observability = ObservabilitySettings(
        langfuse_enabled=True,
        langfuse_base_url="https://trace.invalid",
        langfuse_public_key="synthetic-public",
        langfuse_secret_key="synthetic-secret",  # noqa: S106 -- synthetic credential.
    )
    services = render(deployment_settings).services
    api = json.loads(services["api"].environment["IP_OBSERVABILITY"])
    assert api["langfuse_enabled"] is True
    assert api["langfuse_secret_key"] == "synthetic-secret"  # noqa: S105 -- synthetic credential.
    for name in ("postgres", "migrate", "mcp"):
        assert "synthetic-secret" not in services[name].model_dump_json()
        assert "IP_OBSERVABILITY" not in services[name].environment


def test_disabled_tracing_preserves_null_credentials(
    deployment_settings: DeploymentSettings,
) -> None:
    api = json.loads(render(deployment_settings).services["api"].environment["IP_OBSERVABILITY"])
    assert api["langfuse_enabled"] is False
    assert api["langfuse_secret_key"] is None


def test_sanity_settings_only_reach_api(deployment_settings: DeploymentSettings) -> None:
    deployment_settings.data_agent = DataAgentSettings(
        sanity=SanitySettings(money_columns=["gmv"], expected_max_rows=1)
    )
    services = render(deployment_settings).services
    api = DataAgentSettings.model_validate_json(services["api"].environment["IP_DATA_AGENT"])
    assert api == deployment_settings.data_agent
    for name in ("postgres", "migrate", "mcp"):
        assert "IP_DATA_AGENT" not in services[name].environment


def test_foreign_settings_not_loaded(
    deployment_settings: DeploymentSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "foreign")
    monkeypatch.setenv("POSTGRES_PASSWORD", "foreign")
    monkeypatch.setenv("IP_LLM__API_KEY", "foreign")
    monkeypatch.setenv("PGPASSWORD", "foreign")
    config = render(deployment_settings)
    assert "foreign" not in config.model_dump_json()
    env = process_environment(deployment_settings)
    assert "PGPASSWORD" not in env
    assert "IP_LLM__API_KEY" not in env


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        (Command.DOWN, ["-v"]),
        (Command.DOWN, ["--volumes"]),
        (Command.DOWN, ["--rmi", "all"]),
        (Command.UP, ["--force-recreate"]),
        (Command.UP, ["-p", "pathfinder"]),
        (Command.UP, ["--project-name=pathfinder"]),
        (Command.CONFIG, ["-f", "/foreign/compose.yaml"]),
        (Command.CONFIG, ["--env-file", "/foreign/.env"]),
        (Command.EXEC, ["--privileged", "postgres", "true"]),
        (Command.EXEC, ["foreign", "true"]),
    ],
)
def test_unsafe_arguments_rejected(command: Command, arguments: list[str]) -> None:
    with pytest.raises(DeploymentError):
        validate_arguments(Invocation(command=command, arguments=arguments))


def test_missing_secret_fails_without_echoing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in tuple(os.environ):
        if key.startswith("IP_"):
            monkeypatch.delenv(key)
    config = tmp_path / ".env.deployment"
    config.write_text("IP_POSTGRES_SUPERUSER_PASSWORD=private-value\n")
    with pytest.raises(ValidationError) as error:
        DeploymentSettings(_env_file=config)
    assert "bootstrap" in str(error.value)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("port", [0, 65536])
def test_invalid_port_rejected(port: int) -> None:
    with pytest.raises(ValidationError):
        DeploymentSettings(_env_file=ROOT / ".env.deployment.example", db_host_port=port)


def test_unknown_deployment_file_field_rejected(tmp_path: Path) -> None:
    config = tmp_path / ".env.deployment"
    config.write_text((ROOT / ".env.deployment.example").read_text() + "\nIP_TYPO=bad\n")
    with pytest.raises(ValidationError):
        DeploymentSettings(_env_file=config)


def test_secret_repr_is_redacted(deployment_settings: DeploymentSettings) -> None:
    assert deployment_settings.bootstrap.mcp_password.get_secret_value() not in repr(
        deployment_settings
    )
    assert deployment_settings.postgres_superuser_password.get_secret_value() not in str(
        deployment_settings
    )


def test_router_threshold_reaches_only_api(deployment_settings: DeploymentSettings) -> None:
    deployment_settings.router = RouterSettings(min_confidence=0.75)
    config = render(deployment_settings)
    assert json.loads(config.services["api"].environment["IP_ROUTER"]) == {
        "min_confidence": 0.75,
        "strategy": "hybrid",
    }
    assert all(
        "IP_ROUTER" not in service.environment
        for name, service in config.services.items()
        if name != "api"
    )


def test_compose_paths_are_absolute(deployment_settings: DeploymentSettings) -> None:
    call = Invocation(command=Command.CONFIG)
    command = compose_prefix("docker", deployment_settings, call)
    assert command[command.index("-f") + 1] == str(ROOT / "docker-compose.yml")
    assert command[command.index("--env-file") + 1] == str(ROOT / ".env.deployment")


def test_port_conflict_reports_without_killing() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(DeploymentError, match=str(port)):
            verify_port(port)
        assert listener.fileno() >= 0


def test_unowned_volume_is_rejected(
    deployment_settings: DeploymentSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(["insightpilot_pgdata\n", "pathfinder/pgdata\n"])

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(results))

    monkeypatch.setattr("scripts.deployment.run_command", fake_run)
    with pytest.raises(DeploymentError, match="ownership"):
        verify_volume("docker", deployment_settings, {})


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_external_failure_is_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["docker"], 1, stderr="secret")
        return subprocess.CompletedProcess([], 1, stdout="secret", stderr="secret")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(DeploymentError) as error:
        run_command(["docker"], {}, timeout=1)
    assert "secret" not in str(error.value)


def test_make_lifecycle_uses_scoped_wrapper() -> None:
    make = shutil.which("make")
    assert make is not None
    for target in ("up", "down"):
        output = run_command([make, "--no-print-directory", "-n", target], dict(os.environ)).stdout
        assert f'bash "{ROOT}/scripts/compose.sh" --profile core {target}' in output
