"""Explicit local probe lifecycle; no global restart, removal or remote management."""

import argparse
import os
import socket
import subprocess
from enum import StrEnum
from pathlib import Path

from pydantic import DirectoryPath, Field, FilePath

from app.core.settings_base import ConfigModel, ProcessSettings, Secret
from spikes.capacity.contracts import FailureKind, ProbeError
from spikes.capacity.observe import command, current_memtotal

ROOT = Path(__file__).resolve().parents[2]


class DeploymentFields(ConfigModel):
    """Compose interpolation is an explicit role-scoped allowlist."""

    project: str = Field(pattern=r"^insightpilot-test-step08-[a-z0-9-]+$")
    postgres_password: Secret = Field(repr=False)
    minio_user: str = Field(min_length=3)
    minio_password: Secret = Field(repr=False, min_length=8)
    model_token: Secret = Field(repr=False)
    ssh_config: FilePath
    jump_key: FilePath
    target_key: FilePath
    known_hosts: FilePath
    evidence: DirectoryPath

    def environment(self) -> dict[str, str]:
        """Never hand API, production deployment or foreign Compose values to this stack."""
        values = {
            "PROJECT": self.project,
            "POSTGRES_PASSWORD": self.postgres_password.get_secret_value(),
            "MINIO_USER": self.minio_user,
            "MINIO_PASSWORD": self.minio_password.get_secret_value(),
            "MODEL_TOKEN": self.model_token.get_secret_value(),
            "SSH_CONFIG": str(self.ssh_config),
            "JUMP_KEY": str(self.jump_key),
            "TARGET_KEY": str(self.target_key),
            "KNOWN_HOSTS": str(self.known_hosts),
            "EVIDENCE": str(self.evidence),
        }
        return {f"IP_CAPACITY_{name}": value for name, value in values.items()}


class DeploymentSettings(ProcessSettings):
    """Separate settings file for the local probe operator."""

    process_name = "capacity-deployment"
    capacity_deployment: DeploymentFields


class Action(StrEnum):
    """Destructive cleanup and arbitrary Compose overrides are deliberately absent."""

    CONFIG = "config"
    BUILD = "build"
    UP_STORAGE = "up-storage"
    UP_PROBE = "up-probe"
    DATABASE_LOAD = "database-load"
    STOP = "stop"
    PS = "ps"
    STATS = "stats"


def verify_port(port: int) -> None:
    """Name Linux listeners and leave them running when a loopback port conflicts."""
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            owner = command(["ss", "-ltnp", f"sport = :{port}"])
            raise ProbeError(
                FailureKind.PREREQUISITE, f"127.0.0.1:{port} occupied: {owner}"
            ) from exc


def compose_args(settings: DeploymentFields, action: Action) -> list[str]:
    """Always pass an explicit project/file; never allow positional option injection."""
    prefix = [
        "docker",
        "compose",
        "-p",
        settings.project,
        "--project-directory",
        str(ROOT),
        "--env-file",
        "/dev/null",
        "-f",
        str(ROOT / "spikes/capacity/compose.yml"),
    ]
    commands = {
        Action.CONFIG: ["--profile", "probe", "config", "--quiet"],
        Action.BUILD: ["--profile", "probe", "build"],
        Action.UP_STORAGE: [
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "180",
            "postgres",
            "etcd",
            "minio",
            "milvus",
        ],
        Action.UP_PROBE: ["--profile", "probe", "up", "-d", "model-tunnel", "workload"],
        Action.DATABASE_LOAD: [
            "--profile",
            "probe",
            "run",
            "--no-deps",
            "-d",
            "workload",
            "python",
            "-m",
            "spikes.capacity.database_probe",
        ],
        Action.STOP: ["--profile", "probe", "stop", "--timeout", "10"],
        Action.PS: ["--profile", "probe", "ps", "-a"],
        Action.STATS: ["--profile", "probe", "stats", "--no-stream"],
    }
    return prefix + commands[action]


def main() -> int:
    """Execute one authorized local action and preserve all artifacts and volumes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", type=Action, choices=list(Action))
    action = parser.parse_args().action
    settings = DeploymentSettings.load().capacity_deployment
    # Admission floor only, never reported as measured acceptance.
    if action in (Action.UP_STORAGE, Action.UP_PROBE) and current_memtotal() < 4 * 1024**3:
        raise ProbeError(FailureKind.HEADROOM, "Probe startup requires at least 4 GiB allocated.")
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("IP_", "COMPOSE_"))}
    environment.update(settings.environment())
    environment["COMPOSE_DISABLE_ENV_FILE"] = "1"
    try:
        result = subprocess.run(  # noqa: S603 -- validated action and isolated project.
            compose_args(settings, action), env=environment, timeout=1200, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(FailureKind.TIMEOUT, "Probe lifecycle command timed out.") from exc
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
