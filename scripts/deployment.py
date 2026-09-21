"""Typed, project-scoped PostgreSQL operator commands (not application Settings)."""

import argparse
import json
import os
import shutil
import socket
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import structlog
from pydantic import BaseModel, Field, SecretStr, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config_models import (
    DataAgentSettings,
    HTTPSettings,
    ObservabilitySettings,
    RouterSettings,
)
from app.core.errors import InsightPilotError
from app.core.llm_config import ModelRole, ModelRoleSettings
from app.retrieval.config import RetrievalSettings
from scripts.model_evidence import Provenance

ROOT = Path(__file__).resolve().parents[1]
LOGGER = structlog.get_logger()
PROFILES = ("core", "retrieval", "full", "dev", "ui")
Secret = Annotated[SecretStr, Field(min_length=1)]


class DeploymentError(InsightPilotError):
    """An operator command failed; raw subprocess output may contain secrets."""

    code = "DEPLOYMENT_FAILED"
    user_message = "Deployment failed; inspect the selected project's local service status."


class BootstrapSecrets(BaseModel):
    """Runtime credentials delivered only to the database bootstrap environment."""

    model_config = {"extra": "forbid"}
    app_password: Secret
    etl_password: Secret
    mcp_password: Secret


class MinioSettings(BaseModel):
    """Object-store credentials stay in the deployment process and storage services."""

    model_config = {"extra": "forbid", "hide_input_in_errors": True}
    user: Annotated[SecretStr, Field(min_length=3)] | None = Field(default=None, repr=False)
    password: Annotated[SecretStr, Field(min_length=8)] | None = Field(default=None, repr=False)


class DeploymentSettings(BaseSettings):
    """The complete deployment allowlist; unrelated shell variables are ignored."""

    model_config = SettingsConfigDict(
        env_prefix="IP_", env_nested_delimiter="__", extra="forbid", hide_input_in_errors=True
    )
    compose_project_name: str = Field(
        default="insightpilot", pattern=r"^insightpilot(?:-[a-z0-9][a-z0-9_-]*)?$"
    )
    db_host_port: int = Field(default=15432, ge=1024, le=65535)
    postgres_superuser_password: Secret
    bootstrap: BootstrapSecrets
    api_host_port: int = Field(default=18000, ge=1024, le=65535)
    mcp_auth_token: Secret | None = None
    jwt_secret: Secret | None = None
    llm_base_url: str = "https://example.invalid/v1"
    llm_model: str = "qwen3.6-flash-2026-04-16"
    llm_timeout_s: float = Field(default=45, ge=0.01, le=120)
    llm_roles: dict[ModelRole, ModelRoleSettings] = Field(default_factory=dict)
    llm_capabilities_paths: list[str] = Field(
        default_factory=lambda: ["app/resources/provider_capabilities.json"],
        min_length=1,
        max_length=16,
    )
    llm_api_key: Secret | None = None
    data_agent: DataAgentSettings = Field(default_factory=DataAgentSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings(log_format="json")
    )

    def compose_environment(self) -> dict[str, str]:
        """Expose only declared deployment values for explicit Compose interpolation."""
        observation = self.observability.model_dump(mode="json")
        for field in ("langfuse_public_key", "langfuse_secret_key"):
            secret = getattr(self.observability, field)
            observation[field] = secret.get_secret_value() if secret is not None else None
        return {
            "IP_RETRIEVAL": self.retrieval.model_dump_json(),
            "IP_MINIO__USER": self.minio.user.get_secret_value() if self.minio.user else "",
            "IP_MINIO__PASSWORD": self.minio.password.get_secret_value()
            if self.minio.password
            else "",
            "IP_DATA_AGENT": self.data_agent.model_dump_json(),
            "IP_ROUTER": self.router.model_dump_json(),
            "IP_HTTP": self.http.model_dump_json(),
            "IP_OBSERVABILITY": json.dumps(observation),
            "IP_API_HOST_PORT": str(self.api_host_port),
            "IP_MCP_AUTH_TOKEN": self.mcp_auth_token.get_secret_value()
            if self.mcp_auth_token
            else "",
            "IP_JWT_SECRET": self.jwt_secret.get_secret_value() if self.jwt_secret else "",
            "IP_LLM_BASE_URL": self.llm_base_url,
            "IP_LLM_MODEL": self.llm_model,
            "IP_LLM_TIMEOUT_S": str(self.llm_timeout_s),
            "IP_LLM_ROLES": TypeAdapter(dict[ModelRole, ModelRoleSettings])
            .dump_json(self.llm_roles)
            .decode(),
            "IP_LLM_CAPABILITIES_PATHS": TypeAdapter(list[str])
            .dump_json(self.llm_capabilities_paths)
            .decode(),
            "IP_LLM_API_KEY": self.llm_api_key.get_secret_value() if self.llm_api_key else "",
            "IP_COMPOSE_PROJECT_NAME": self.compose_project_name,
            "IP_DB_HOST_PORT": str(self.db_host_port),
            "IP_POSTGRES_SUPERUSER_PASSWORD": self.postgres_superuser_password.get_secret_value(),
            "IP_BOOTSTRAP__APP_PASSWORD": self.bootstrap.app_password.get_secret_value(),
            "IP_BOOTSTRAP__ETL_PASSWORD": self.bootstrap.etl_password.get_secret_value(),
            "IP_BOOTSTRAP__MCP_PASSWORD": self.bootstrap.mcp_password.get_secret_value(),
        }


class Command(StrEnum):
    """Supported operations; destructive reset has no command here."""

    RUN = "run"
    UP = "up"
    DOWN = "down"
    CONFIG = "config"
    PS = "ps"
    LOGS = "logs"
    STATS = "stats"
    EXEC = "exec"
    BOOTSTRAP = "bootstrap"


class Invocation(BaseModel):
    """Validated wrapper arguments, distinct from service configuration."""

    dev: bool = False
    profiles: list[str] = Field(default_factory=lambda: ["core"])
    deployment_file: Path = ROOT / ".env.deployment"
    command: Command
    arguments: list[str] = Field(default_factory=list)


def process_environment(settings: DeploymentSettings) -> dict[str, str]:
    """Do not inherit foreign Compose selectors, dotenv values or PG credentials."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COMPOSE_", "IP_", "PG"))
    }
    env.update(settings.compose_environment())
    env["COMPOSE_DISABLE_ENV_FILE"] = "1"
    return env


def run_command(
    command: list[str], env: dict[str, str], *, timeout: int = 30, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Bound operator I/O; failures are never automatically retried or echoed."""
    try:
        result = subprocess.run(  # noqa: S603 -- structured executable/args, never shell evaluation.
            command,
            cwd=ROOT,
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeploymentError("Operator command unavailable or timed out.") from exc
    if result.returncode:
        raise DeploymentError(
            "Operator command returned a failure exit code.", diagnostics=result.stderr
        )
    return result


def compose_prefix(docker: str, settings: DeploymentSettings, call: Invocation) -> list[str]:
    """Anchor every operation to this repository and one isolated project."""
    command = [
        docker,
        "compose",
        "-p",
        settings.compose_project_name,
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(call.deployment_file),
        "-f",
        str(ROOT / "docker-compose.yml"),
    ]
    if call.dev:
        command.extend(["-f", str(ROOT / "docker-compose.dev.yml")])
    for profile in call.profiles:
        command.extend(["--profile", profile])
    return command


def validate_arguments(call: Invocation) -> None:
    """Only permit documented flags; forwarded exec arguments follow postgres."""
    allowed = {
        Command.UP: {"-d", "--detach", "--wait", "--build", "postgres", "mcp", "api", "migrate"},
        Command.DOWN: set(),
        Command.CONFIG: {"--format", "json", "--services", "--quiet"},
        Command.PS: {"-a", "--all", "--format", "json", "postgres", "mcp", "api", "migrate"},
        Command.LOGS: {"--no-color", "postgres", "mcp", "api", "migrate"},
        Command.STATS: {"--no-stream", "postgres", "mcp", "api"},
        Command.BOOTSTRAP: set(),
    }
    if call.command is Command.RUN:
        if call.arguments[:2] == ["--rm", "model-diagnostics"]:
            validate_model_diagnostic(call.arguments[2:])
            return
        if call.arguments != ["--rm", "seed"]:
            raise DeploymentError("run supports only --rm seed.")
        return
    if call.command is Command.EXEC:
        args = call.arguments
        if args[:1] == ["-T"]:
            args = args[1:]
        if len(args) < 2 or args[0] not in {"postgres", "api", "mcp", "migrate"}:  # noqa: PLR2004 -- service plus executable.
            raise DeploymentError("exec requires [-T] SERVICE COMMAND [ARGS].")
        return
    for command in (Command.UP, Command.PS, Command.LOGS, Command.STATS):
        allowed[command].update({"etcd", "minio", "milvus", "model-tunnel"})
    if any(argument not in allowed[call.command] for argument in call.arguments):
        raise DeploymentError(
            "Unsupported arguments; project overrides and volume deletion are forbidden."
        )


def validate_model_diagnostic(arguments: list[str]) -> None:
    """Allow a bounded provenance value without exposing arbitrary Docker arguments."""
    if arguments == ["python", "-m", "scripts.probe_model_runtime", "--ready"]:
        return
    prefix = ["python", "-m", "scripts.bench_model_runtime", "--provenance-json"]
    if arguments[:-1] != prefix:
        raise DeploymentError("Model diagnostics require a supported command and provenance.")
    try:
        Provenance.model_validate_json(arguments[-1])
    except ValidationError as exc:
        raise DeploymentError("Invalid model measurement provenance.") from exc


def verify_volume(docker: str, settings: DeploymentSettings, env: dict[str, str]) -> None:
    """Reject a colliding volume unless Compose labels establish ownership."""
    name = f"{settings.compose_project_name}_pgdata"
    names = run_command([docker, "volume", "ls", "--format", "{{.Name}}"], env).stdout.splitlines()
    if name not in names:
        return
    labels = run_command(
        [
            docker,
            "volume",
            "inspect",
            "--format",
            '{{index .Labels "com.docker.compose.project"}}/{{index .Labels "com.docker.compose.volume"}}',
            name,
        ],
        env,
    ).stdout.strip()
    if labels != f"{settings.compose_project_name}/pgdata":
        raise DeploymentError(
            "The selected data volume lacks matching InsightPilot ownership labels."
        )


def verify_port(port: int) -> None:
    """Check WSL and Windows loopback listeners without stopping any owner."""
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise DeploymentError(
                f"127.0.0.1:{port} is occupied; the listener was not changed."
            ) from exc
    powershell = shutil.which("powershell.exe")
    if powershell:
        result = run_command(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -eq {port} | Select-Object -ExpandProperty OwningProcess",
            ],
            dict(os.environ),
        )
        if result.stdout.strip():
            raise DeploymentError(f"Windows port {port} is occupied; the listener was not changed.")


def preflight(docker: str, settings: DeploymentSettings, call: Invocation) -> None:
    """Check minimum bootstrap capacity and isolation; full-stack peaks are Step 0.8."""
    env = process_environment(settings)
    memory = int(run_command([docker, "info", "--format", "{{.MemTotal}}"], env).stdout)
    if memory < (1536 + 1024) * 1024 * 1024:
        raise DeploymentError("Docker needs the 1536 MiB database limit plus 1 GiB headroom.")
    verify_volume(docker, settings, env)
    running = run_command(
        [
            docker,
            "ps",
            "--filter",
            f"label=com.docker.compose.project={settings.compose_project_name}",
            "--filter",
            "label=com.docker.compose.service=postgres",
            "--format",
            "{{.ID}}",
        ],
        env,
    ).stdout.strip()
    selected = set(call.arguments) & {
        "postgres",
        "migrate",
        "mcp",
        "api",
        "seed",
        "model-tunnel",
        "model-diagnostics",
        "etcd",
        "minio",
        "milvus",
    }
    starts_storage = bool(selected & {"minio", "milvus"}) or (
        not selected and bool(set(call.profiles) & {"retrieval", "full", "dev", "ui"})
    )
    if starts_storage and not (settings.minio.user and settings.minio.password):
        raise DeploymentError("Milvus startup requires independent MinIO credentials.")
    starts_api = not selected or "api" in selected
    starts_mcp = starts_api or "mcp" in selected
    if starts_mcp and not settings.mcp_auth_token:
        raise DeploymentError("MCP startup requires its bearer secret in deployment settings.")
    if starts_api:
        if not all((settings.jwt_secret, settings.llm_api_key)):
            raise DeploymentError(
                "API startup requires JWT and LLM secrets in deployment settings."
            )
        api_running = run_command(
            [
                docker,
                "ps",
                "--filter",
                f"label=com.docker.compose.project={settings.compose_project_name}",
                "--filter",
                "label=com.docker.compose.service=api",
                "--format",
                "{{.ID}}",
            ],
            env,
        ).stdout.strip()
        if not api_running:
            verify_port(settings.api_host_port)
    if call.dev and not running:
        verify_port(settings.db_host_port)


def execute(docker: str, settings: DeploymentSettings, call: Invocation) -> str:
    """Run one validated operation; SQL failures stop bootstrap without resetting data."""
    validate_arguments(call)
    prefix = compose_prefix(docker, settings, call)
    env = process_environment(settings)
    selected = set(call.arguments) & {
        "postgres",
        "migrate",
        "mcp",
        "api",
        "seed",
        "etcd",
        "minio",
        "milvus",
        "model-tunnel",
        "model-diagnostics",
    }
    uses_model = bool(selected & {"model-tunnel", "model-diagnostics"}) or (
        not selected and bool(set(call.profiles) & {"retrieval", "full", "dev", "ui"})
    )
    if uses_model and call.command in {Command.UP, Command.RUN}:
        from scripts.tunnel_deployment import prepare  # noqa: PLC0415 -- operator modules share DeploymentError.

        env.update(prepare(ROOT, write_config=call.command is Command.UP))
        env["IP_RETRIEVAL"] = settings.retrieval.model_copy(
            update={"enabled": True}
        ).model_dump_json()

    if call.command in {Command.UP, Command.RUN}:
        preflight(docker, settings, call)
    if call.command is Command.BOOTSTRAP:
        # Compose forwards values from the client environment for bare -e keys.
        # This applies edited runtime passwords without recreating the container.
        env.update(
            {
                "IP_BOOTSTRAP_APP_PASSWORD": settings.bootstrap.app_password.get_secret_value(),
                "IP_BOOTSTRAP_ETL_PASSWORD": settings.bootstrap.etl_password.get_secret_value(),
                "IP_BOOTSTRAP_MCP_PASSWORD": settings.bootstrap.mcp_password.get_secret_value(),
            }
        )
        command = [
            *prefix,
            "exec",
            "-T",
            "-e",
            "IP_BOOTSTRAP_APP_PASSWORD",
            "-e",
            "IP_BOOTSTRAP_ETL_PASSWORD",
            "-e",
            "IP_BOOTSTRAP_MCP_PASSWORD",
            "postgres",
            "psql",
            "-X",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            "/docker-entrypoint-initdb.d/01_bootstrap.sql",
        ]
        run_command(command, env, timeout=120)
        LOGGER.info("database_bootstrap_complete", project=settings.compose_project_name)
        return ""
    result = run_command([*prefix, call.command.value, *call.arguments], env, timeout=900)
    return result.stdout + result.stderr


def parse_invocation() -> Invocation:
    """Parse wrapper flags before the command; never pass global overrides through."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--deployment-file", type=Path, default=ROOT / ".env.deployment")
    parser.add_argument("command", choices=list(Command))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = (ROOT / args.deployment_file).resolve()
    if path.parent != ROOT or not path.name.startswith(".env.") or not path.is_file():
        parser.error(
            "--deployment-file must be an existing root-local .env.* file (default .env.deployment)"
        )
    return Invocation(
        dev=args.dev,
        profiles=args.profile or ["core"],
        deployment_file=path,
        command=Command(args.command),
        arguments=args.arguments,
    )


def main() -> int:
    """Print sanitized operator failures, never settings values or subprocess traces."""
    call = parse_invocation()
    try:
        settings = DeploymentSettings(_env_file=call.deployment_file)
        docker = shutil.which("docker")
        if docker is None:
            raise DeploymentError("Docker is required.")
        output = execute(docker, settings, call)
    except ValidationError as exc:
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        LOGGER.exception("deployment_configuration_invalid", fields=fields, exc_info=False)
        return 2
    except DeploymentError as exc:
        LOGGER.exception("deployment_failed", code=exc.code, message=str(exc), exc_info=False)
        return 1
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
