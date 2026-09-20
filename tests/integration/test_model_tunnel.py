"""Real isolated double-hop SSH; no company keys, host access or live GPU."""

import json
import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.schemas.model_runtime import EmbedMode, ModelFailureKind
from model_runtime.errors import ModelError
from model_tunnel.config import TunnelSettings
from model_tunnel.render import render
from tests.fakes.model_endpoint import TOKEN
from tests.fakes.model_runtime import FakeModels

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def command(args: list[str], *, timeout: int = 90) -> str:
    result = subprocess.run(  # noqa: S603 -- synthetic isolated fixture arguments.
        args, capture_output=True, text=True, timeout=timeout, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout.strip()


@pytest.fixture(scope="module")
def ssh_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, str, str]]:
    assert shutil.which("docker")
    assert shutil.which("ssh-keygen")
    root = tmp_path_factory.mktemp("model-ssh")
    prefix = "insightpilot-test-ssh-" + uuid4().hex[:10]
    for name in ("client", "wrong", "jump", "target"):
        command(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)])
    (root / "authorized_keys").write_text((root / "client.pub").read_text())
    (root / "model_endpoint.py").write_text((ROOT / "tests/fakes/model_endpoint.py").read_text())
    (root / "model_metadata.json").write_text(FakeModels().metadata().model_dump_json())
    (root / "sshd_config").write_text("""Port 2222
ListenAddress 0.0.0.0
PidFile /tmp/ip-sshd.pid
AuthorizedKeysFile /fixture/authorized_keys
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AllowUsers appuser
AllowTcpForwarding yes
LogLevel ERROR
""")
    (root / "Dockerfile").write_text("""FROM python:3.12.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server && useradd --uid 10001 --create-home appuser && passwd -d appuser && mkdir -p /run/sshd
COPY --chown=10001:10001 . /fixture/
RUN chmod 600 /fixture/jump /fixture/target && chmod 644 /fixture/authorized_keys
WORKDIR /fixture
USER appuser
""")
    command(["docker", "build", "-t", prefix + "-server", str(root)], timeout=180)
    command(
        ["docker", "build", "-t", prefix + "-client", "-f", "docker/Dockerfile.model-tunnel", "."],
        timeout=180,
    )
    command(["docker", "network", "create", prefix])
    containers = []
    try:
        for role in ("jump", "target"):
            name = prefix + "-" + role
            command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "--network",
                    prefix,
                    "--network-alias",
                    role,
                    "--memory",
                    "128m",
                    prefix + "-server",
                    "sh",
                    "-c",
                    "python /fixture/model_endpoint.py >/dev/null 2>&1 & exec /usr/sbin/sshd -D -e -f /fixture/sshd_config -h /fixture/"
                    + role,
                ]
            )
            containers.append(name)
        known = root / "known_hosts"
        known.write_text(
            "".join(
                "[" + role + "]:2222 " + (root / (role + ".pub")).read_text()
                for role in ("jump", "target")
            )
        )
        config = TunnelSettings(
            jump_host="jump",
            jump_user="appuser",
            jump_port=2222,
            target_host="target",
            target_user="appuser",
            target_port=2222,
            jump_key_path=root / "client",
            target_key_path=root / "client",
            known_hosts_path=known,
            config_path=root / "config",
        )
        (root / "config").write_text(render(config))
        containers.append(prefix + "-client")
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                prefix + "-client",
                "--network",
                prefix,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--memory",
                "128m",
                "-p",
                "127.0.0.1::8100",
                "--mount",
                f"type=bind,src={root / 'config'},dst=/run/ip-ssh/config,readonly",
                "--mount",
                f"type=bind,src={root / 'client'},dst=/run/ip-ssh/jump_key,readonly",
                "--mount",
                f"type=bind,src={root / 'client'},dst=/run/ip-ssh/target_key,readonly",
                "--mount",
                f"type=bind,src={known},dst=/run/ip-ssh/known_hosts,readonly",
                prefix + "-client",
            ]
        )
        binding = json.loads(command(["docker", "inspect", prefix + "-client"]))[0][
            "NetworkSettings"
        ]["Ports"]["8100/tcp"][0]
        assert binding["HostIp"] == "127.0.0.1"
        url = "http://127.0.0.1:" + binding["HostPort"]
        wait_healthy(url)
        yield root, prefix, url
    finally:
        evidence = ROOT / "model-tunnel-evidence"
        evidence.mkdir(exist_ok=True)
        failures = []
        for name in reversed(containers):
            try:
                logs = subprocess.run(  # noqa: S603 -- owned synthetic fixture containers.
                    [shutil.which("docker") or "/usr/bin/docker", "logs", name],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                (evidence / (name + ".log")).write_bytes(logs.stdout + logs.stderr)
                command(["docker", "stop", "--time", "5", name], timeout=20)
            except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Isolated SSH finalization failed", failures)
        # Containers, network, images and synthetic fixture files are retained.


def endpoint_healthy(client: httpx.Client, url: str) -> bool:
    try:
        return client.get(url + "/health").json() == {"status": "ok"}
    except (httpx.HTTPError, ValueError):
        return False


def wait_healthy(url: str) -> None:
    until = time.monotonic() + 65
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < until:
            if endpoint_healthy(client, url):
                return
            time.sleep(1)
    raise AssertionError("Isolated SSH forwarding failed to recover")


def test_double_hop_forwarded_health_and_readonly_selected_mounts(
    ssh_stack: tuple[Path, str, str],
) -> None:
    _, prefix, url = ssh_stack
    wait_healthy(url)
    inspected = json.loads(command(["docker", "inspect", prefix + "-client"]))[0]
    assert len(inspected["Mounts"]) == 4  # noqa: PLR2004 -- four individually selected SSH inputs.
    assert all(not item["RW"] for item in inspected["Mounts"])
    assert inspected["Config"]["User"] != "0:0"


def test_dropped_target_connection_recovers_without_shared_host_faults(
    ssh_stack: tuple[Path, str, str],
) -> None:
    _, prefix, url = ssh_stack
    command(["docker", "restart", "--time", "2", prefix + "-target"])
    wait_healthy(url)


def test_unreachable_model_endpoint_is_not_healthy(ssh_stack: tuple[Path, str, str]) -> None:
    _, prefix, url = ssh_stack
    command(["docker", "stop", "--time", "2", prefix + "-target"])
    try:
        with httpx.Client(timeout=2, trust_env=False) as client, pytest.raises(httpx.HTTPError):
            client.get(url + "/health").raise_for_status()
    finally:
        command(["docker", "start", prefix + "-target"])
        wait_healthy(url)


@pytest.mark.parametrize("failure", ["host-key", "authentication"])
def test_invalid_trust_or_key_fails_closed(ssh_stack: tuple[Path, str, str], failure: str) -> None:
    root, prefix, _ = ssh_stack
    bad = root / ("bad-" + failure)
    if failure == "host-key":
        bad.write_text("[jump]:2222 " + (root / "wrong.pub").read_text())
        mount = f"type=bind,src={bad},dst=/run/ip-ssh/known_hosts,readonly"
        key = root / "client"
    else:
        mount = f"type=bind,src={root / 'known_hosts'},dst=/run/ip-ssh/known_hosts,readonly"
        key = root / "wrong"
    result = subprocess.run(  # noqa: S603 -- synthetic isolated fixture arguments.
        [
            shutil.which("docker") or "/usr/bin/docker",
            "run",
            "--name",
            prefix + "-bad-" + failure,
            "--network",
            prefix,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={root / 'config'},dst=/run/ip-ssh/config,readonly",
            "--mount",
            f"type=bind,src={key},dst=/run/ip-ssh/jump_key,readonly",
            "--mount",
            f"type=bind,src={key},dst=/run/ip-ssh/target_key,readonly",
            "--mount",
            mount,
            prefix + "-client",
            "ssh",
            "-F",
            "/run/ip-ssh/config",
            "-o",
            "ConnectTimeout=3",
            "ip-model",
            "true",
        ],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert (root / "known_hosts").read_text().startswith("[jump]:2222 ")


async def test_model_client_outage_and_reconnect_over_isolated_tunnel(
    ssh_stack: tuple[Path, str, str],
) -> None:
    _, prefix, url = ssh_stack
    client = ModelRuntimeClient(ModelRuntimeClientSettings(base_url=url, auth_token=TOKEN))
    try:
        assert (await client.ready(deadline=Deadline(time.monotonic() + 2))).ready
        await asyncio.to_thread(command, ["docker", "stop", "--time", "2", prefix + "-target"])
        try:
            with pytest.raises(ModelError) as error:
                await client.embed(["isolated query"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 5))
            assert error.value.kind in {ModelFailureKind.UNAVAILABLE, ModelFailureKind.DEADLINE}
        finally:
            await asyncio.to_thread(command, ["docker", "start", prefix + "-target"])
            await asyncio.to_thread(wait_healthy, url)
        result = await client.embed(["recovered query"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 5))
        assert len(result.dense) == len(result.sparse) == 1
        assert result.batches[0].metadata == FakeModels().metadata()
        assert client.breaker.failures == 0
    finally:
        await client.aclose()
