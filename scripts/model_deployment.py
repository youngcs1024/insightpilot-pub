"""Independent server lifecycle; never called by local Compose or the API."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from model_runtime.config import ModelRuntimeSettings
from scripts.deployment import DeploymentError
from scripts.model_deployment_settings import ModelDeploymentSettings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Manage only the fixed remote model project with validated process inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("up", "stop", "ps", "logs", "config"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    allowed = {
        "up": {"-d", "--wait"},
        "stop": set(),
        "ps": {"-a"},
        "logs": {"--no-color", "--tail", "100"},
        "config": {"--quiet"},
    }
    if any(value not in allowed[args.command] for value in args.arguments):
        parser.error("Only bounded project lifecycle arguments are supported")
    if args.command == "config" and args.arguments != ["--quiet"]:
        parser.error("Only config --quiet is supported; rendered configuration contains secrets")
    deploy = ModelDeploymentSettings.load().model_server
    runtime = ModelRuntimeSettings.load().model_server
    docker = shutil.which("docker")
    if docker is None:
        raise DeploymentError("Docker is unavailable")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("IP_", "COMPOSE_", "PG"))
    }
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "IP_MODEL_SERVER__IMAGE": deploy.image,
            "IP_MODEL_SERVER__GPU_ID": deploy.gpu_id,
            "IP_MODEL_SERVER__CACHE_PATH": str(deploy.cache_path),
        }
    )
    for key, value in runtime.model_dump(exclude={"auth_token"}).items():
        env["IP_MODEL_SERVER__" + key.upper()] = str(value)
    env["IP_MODEL_SERVER__AUTH_TOKEN"] = runtime.auth_token.get_secret_value()
    command = [
        docker,
        "compose",
        "-p",
        "insightpilot-model",
        "--project-directory",
        str(ROOT),
        "--env-file",
        "/dev/null",
        "-f",
        str(ROOT / "docker-compose.model-server.yml"),
        args.command,
        *args.arguments,
    ]
    try:
        result = subprocess.run(command, env=env, timeout=210, capture_output=True, check=False)  # noqa: S603 -- fixed executable, project and allowlisted argv.
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeploymentError("Model lifecycle operation failed or timed out") from exc
    # Compose can echo resolved environment in errors; report status only.
    print(f"Model lifecycle {args.command}: exit={result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
