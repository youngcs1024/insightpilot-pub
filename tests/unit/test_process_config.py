"""Independent model/tunnel configuration contracts with no GPU or SSH access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings_base import PROJECT_ROOT, ProcessSettings
from model_runtime.config import ModelRuntimeSettings
from model_tunnel.config import TunnelProcessSettings
from scripts.model_deployment_settings import ModelDeploymentSettings

QUEUE_CAPACITY = 8
EMBED_MAX_LENGTH = 512
RERANK_MAX_LENGTH = 320
SSH_PORT = 22
TUNNEL_HOST_PORT = 18100


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in os.environ:
        if key.upper().startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(ProcessSettings, "project_root", tmp_path)


@pytest.fixture
def runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_SERVER__AUTH_TOKEN", "runtime-fixture-value")
    monkeypatch.setenv("IP_MODEL_SERVER__EMBED_REVISION", "a" * 40)
    monkeypatch.setenv("IP_MODEL_SERVER__RERANK_REVISION", "b" * 40)


@pytest.fixture
def tunnel_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["JUMP_HOST", "TARGET_HOST"]:
        monkeypatch.setenv(f"IP_TUNNEL__{name}", "ssh.example.invalid")
    for name in ["JUMP_USER", "TARGET_USER"]:
        monkeypatch.setenv(f"IP_TUNNEL__{name}", "fixture-user")
    for name in ["JUMP_KEY_PATH", "TARGET_KEY_PATH", "KNOWN_HOSTS_PATH"]:
        path = tmp_path / name.lower()
        path.write_text("dummy file; no network use")
        monkeypatch.setenv(f"IP_TUNNEL__{name}", str(path))
    monkeypatch.setenv("IP_TUNNEL__CONFIG_PATH", str(tmp_path / "generated_config"))


@pytest.fixture
def deployment_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_SERVER__IMAGE", "example.invalid/model@sha256:" + "a" * 64)
    monkeypatch.setenv("IP_MODEL_SERVER__GPU_ID", "GPU-12345678-1234-1234-1234-123456789abc")
    monkeypatch.setenv("IP_MODEL_SERVER__CACHE_PATH", str(tmp_path))


@pytest.mark.usefixtures("runtime_environment")
def test_runtime_loads_without_api_credentials() -> None:
    model = ModelRuntimeSettings.load().model_server
    assert model.workers == 1
    assert model.queue_capacity == QUEUE_CAPACITY
    assert model.embed_max_length == EMBED_MAX_LENGTH
    assert model.rerank_max_length == RERANK_MAX_LENGTH
    assert "runtime-fixture-value" not in repr(model)


@pytest.mark.usefixtures("runtime_environment")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AUTH_TOKEN", ""),
        ("EMBED_REVISION", "main"),
        ("RERANK_REVISION", "latest"),
        ("DEVICE", "cpu"),
        ("PRECISION", "bf16"),
        ("WORKERS", "2"),
        ("MAX_CONCURRENCY", "0"),
        ("QUEUE_CAPACITY", "9"),
        ("EMBED_BATCH", "17"),
        ("RERANK_BATCH", "0"),
        ("EMBED_MAX_LENGTH", "513"),
        ("RERANK_MAX_LENGTH", "321"),
    ],
)
def test_runtime_invalid_values_rejected(
    field: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(f"IP_MODEL_SERVER__{field}", value)
    with pytest.raises(ValidationError):
        ModelRuntimeSettings.load()


@pytest.mark.usefixtures("runtime_environment")
def test_runtime_allows_sequential_fp32_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_SERVER__PRECISION", "fp32")
    assert ModelRuntimeSettings.load().model_server.precision == "fp32"


@pytest.mark.usefixtures("runtime_environment")
@pytest.mark.parametrize(
    "key", ["IP_MODEL_SERVER__GPU_ID", "IP_DATABASE__APP_PASSWORD", "IP_MODEL_SERVER__WORKER"]
)
def test_runtime_rejects_deployment_api_and_typos(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(key, "must-not-be-printed")
    with pytest.raises(ValidationError) as error:
        ModelRuntimeSettings.load()
    assert "must-not-be-printed" not in str(error.value)


@pytest.mark.usefixtures("tunnel_environment")
def test_tunnel_loads_explicit_paths_and_defaults() -> None:
    model = TunnelProcessSettings.load().tunnel
    assert model.jump_port == SSH_PORT
    assert model.target_port == SSH_PORT
    assert model.host_port == TUNNEL_HOST_PORT
    assert model.jump_key_path.is_file()


@pytest.mark.usefixtures("tunnel_environment")
@pytest.mark.parametrize("field", ["JUMP_KEY_PATH", "TARGET_KEY_PATH", "KNOWN_HOSTS_PATH"])
def test_tunnel_missing_file_rejected(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"IP_TUNNEL__{field}", "/nonexistent-insightpilot-test/file")
    with pytest.raises(ValidationError):
        TunnelProcessSettings.load()


@pytest.mark.usefixtures("tunnel_environment")
@pytest.mark.parametrize("field", ["JUMP_PORT", "TARGET_PORT", "HOST_PORT"])
def test_tunnel_port_bounds(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"IP_TUNNEL__{field}", "65536")
    with pytest.raises(ValidationError):
        TunnelProcessSettings.load()


@pytest.mark.usefixtures("deployment_environment")
def test_deployment_inputs_are_independent() -> None:
    model = ModelDeploymentSettings.load().model_server
    assert model.cache_path.is_dir()
    assert model.image.endswith("a" * 64)


@pytest.mark.usefixtures("deployment_environment")
@pytest.mark.parametrize(
    ("field", "value"),
    [("IMAGE", "example.invalid/model:latest"), ("GPU_ID", "0"), ("CACHE_PATH", "/missing/cache")],
)
def test_invalid_deployment_inputs(field: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"IP_MODEL_SERVER__{field}", value)
    with pytest.raises(ValidationError):
        ModelDeploymentSettings.load()


@pytest.mark.usefixtures("deployment_environment")
def test_deployment_rejects_runtime_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_SERVER__AUTH_TOKEN", "must-not-be-printed")
    with pytest.raises(ValidationError):
        ModelDeploymentSettings.load()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("model_runtime.config", "ModelRuntimeSettings"),
        ("model_tunnel.config", "TunnelProcessSettings"),
        ("scripts.model_deployment_settings", "ModelDeploymentSettings"),
    ],
)
def test_process_imports_do_not_initialize_api_or_gpu(module: str, name: str) -> None:
    code = (
        "import importlib,sys; "
        f"module = importlib.import_module({module!r}); "
        f"assert hasattr(module, {name!r}); "
        "assert 'app.core.config' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'FlagEmbedding' not in sys.modules; print('isolated')"
    )
    result = subprocess.run(  # noqa: S603 -- fixed module names, no network or service execution.
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
