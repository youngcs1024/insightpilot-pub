"""The production remote network shape must really publish its loopback listener."""

import json
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import yaml

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def command(arguments: list[str], *, timeout: int = 120) -> str:
    result = subprocess.run(  # noqa: S603 -- fixed executable and isolated test project arguments.
        arguments, capture_output=True, text=True, timeout=timeout, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_remote_model_network_publishes_loopback_port(tmp_path: Path) -> None:
    docker = shutil.which("docker")
    assert docker is not None
    production = yaml.safe_load((ROOT / "docker-compose.model-server.yml").read_text())
    project = "insightpilot-model-test-port-" + uuid4().hex[:10]
    compose = tmp_path / "compose.yml"
    compose.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "model-runtime": {
                        "image": "python:3.12.13-slim-bookworm",
                        "user": "10001:10001",
                        "command": ["python", "-m", "http.server", "8100"],
                        "ports": ["127.0.0.1::8100"],
                        "networks": production["services"]["model-runtime"]["networks"],
                        "deploy": {"resources": {"limits": {"memory": "64M", "cpus": "0.2"}}},
                    }
                },
                "networks": production["networks"],
            }
        )
    )
    prefix = [docker, "compose", "-p", project, "--env-file", "/dev/null", "-f", str(compose)]
    try:
        command([*prefix, "up", "-d"])
        identity = command([*prefix, "ps", "-q", "model-runtime"]).strip()
        inspected = json.loads(command([docker, "inspect", identity]))[0]
        bindings = inspected["NetworkSettings"]["Ports"].get("8100/tcp")
        assert bindings, "Internal-only networks silently suppress the host port binding"
        assert len(bindings) == 1
        assert bindings[0]["HostIp"] == "127.0.0.1"
        url = "http://127.0.0.1:" + bindings[0]["HostPort"]
        until = time.monotonic() + 10
        with httpx.Client(timeout=2, trust_env=False) as client:
            while time.monotonic() < until:
                try:
                    if client.get(url).status_code == httpx.codes.OK:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        pytest.fail("The remote model network did not deliver loopback HTTP traffic")
    finally:
        command([*prefix, "stop", "--timeout", "5"], timeout=30)
        # Retain the isolated containers and networks; no shared daemon cleanup.
