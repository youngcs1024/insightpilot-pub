"""ETL config is isolated and source-bound, with no runtime owner credential."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings_base import PROJECT_ROOT, environment_keys
from scripts import deployment
from scripts.deployment import Command, DeploymentError, Invocation, validate_arguments
from scripts.seed_settings import SeedSettings


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in os.environ:
        if name.upper().startswith("IP_"):
            monkeypatch.delenv(name)


def test_seed_environment_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_SEED__PASSWORD", "test-only-secret")
    settings = SeedSettings(_env_file=None)
    assert settings.seed.user == "etl_rw"
    assert "test-only-secret" not in repr(settings)
    source = (PROJECT_ROOT / ".env.seed.example").read_text()
    leaves = {key for key in environment_keys(SeedSettings) if key != "IP_SEED"}
    assert all(key + "=" in source for key in leaves)


def test_seed_rejects_foreign_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_SEED__PASSWORD", "test-only-secret")
    monkeypatch.setenv("IP_DATABASE__APP_PASSWORD", "foreign")
    with pytest.raises(ValidationError):
        SeedSettings(_env_file=None)


def test_seed_process_file_is_project_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SeedSettings, "project_root", tmp_path)
    (tmp_path / ".env.seed").write_text("IP_SEED__PASSWORD=seed-test-only\n")
    assert SeedSettings.load().seed.password.get_secret_value() == "seed-test-only"


@pytest.mark.parametrize(
    "fields", [{"user": "postgres"}, {"batch_size": 0}, {"command_timeout_s": 0}]
)
def test_seed_bounds(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SeedSettings(_env_file=None, seed={"password": "test-only", **fields})


def test_compose_run_is_scoped() -> None:
    validate_arguments(Invocation(command=Command.RUN, arguments=["--rm", "seed"]))
    with pytest.raises(DeploymentError):
        validate_arguments(
            Invocation(command=Command.RUN, arguments=["--rm", "--volume", "/:/host", "seed"])
        )


def test_seed_run_checks_volume_ownership_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = deployment.DeploymentSettings(_env_file=PROJECT_ROOT / ".env.deployment.example")

    def reject(*args: object) -> None:
        raise DeploymentError("ownership mismatch")

    monkeypatch.setattr(deployment, "preflight", reject)
    with pytest.raises(DeploymentError, match="ownership mismatch"):
        deployment.execute(
            "docker", settings, Invocation(command=Command.RUN, arguments=["--rm", "seed"])
        )
