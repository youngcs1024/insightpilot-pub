"""Static remote/local boundary acceptance does not connect to company services."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.check_deployment_contracts import document_issues, load_compose
from scripts.deployment_contracts import environment_issues
from scripts.model_deployment_settings import ModelDeploymentFields

ROOT = Path(__file__).resolve().parents[2]


def test_remote_contract_is_independent_and_rejects_unknown_services() -> None:
    remote = load_compose(ROOT / "docker-compose.model-server.yml")
    assert document_issues(remote, remote=True) == []
    assert document_issues(remote)
    remote.services["api"] = next(iter(remote.services.values()))
    assert "unknown service: api" in document_issues(remote, remote=True)


def test_tunnel_accepts_no_process_credentials() -> None:
    assert environment_issues("model-tunnel", set()).passed
    assert not environment_issues("model-tunnel", {"IP_MODEL_SERVER__AUTH_TOKEN"}).passed


def test_no_default_host_tunnel_port_and_remote_loopback_only() -> None:
    local = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    tunnel = local["services"]["model-tunnel"]
    assert "ports" not in tunnel
    assert "network_mode" not in tunnel
    assert "core" not in tunnel["profiles"]
    assert set(tunnel["networks"]) == {"models", "egress"}
    assert all(
        mount["read_only"] and not mount["bind"]["create_host_path"] for mount in tunnel["volumes"]
    )
    remote = yaml.safe_load((ROOT / "docker-compose.model-server.yml").read_text())
    service = remote["services"]["model-runtime"]
    assert service["ports"] == ["127.0.0.1:8100:8100"]
    assert service["user"] == "10001:10001"
    assert service["volumes"][0]["read_only"]


def test_immutable_local_image_id_and_repository_digest_supported(tmp_path: Path) -> None:
    values = {"gpu_id": "GPU-ec35358d-4afa-1765-2f19-5d2fe68ede13", "cache_path": tmp_path}
    for image in ("sha256:" + "a" * 64, "registry/model@sha256:" + "b" * 64):
        assert ModelDeploymentFields(image=image, **values).image == image
    with pytest.raises(ValidationError):
        ModelDeploymentFields(image="model:latest", **values)
