"""CI must fail on missing infrastructure and never own an external service lifecycle."""

import subprocess
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tests.ci_settings import InfrastructureSettings
from tests.shared_database import pg_container, require_docker


@pytest.fixture(autouse=True)
def isolated_infrastructure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSIGHTPILOT_TEST_POSTGRES", raising=False)
    for name in ("HOST", "PORT", "PASSWORD", "APP_PASSWORD"):
        monkeypatch.delenv("INSIGHTPILOT_TEST_POSTGRES__" + name, raising=False)
    monkeypatch.delenv("INSIGHTPILOT_TEST_REQUIRE_DOCKER", raising=False)


def test_external_postgres_is_explicit_and_secrets_are_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert InfrastructureSettings().postgres is None
    monkeypatch.setenv("INSIGHTPILOT_TEST_POSTGRES__PORT", "5432")
    monkeypatch.setenv("INSIGHTPILOT_TEST_POSTGRES__PASSWORD", "ci-admin-only")
    monkeypatch.setenv("INSIGHTPILOT_TEST_POSTGRES__APP_PASSWORD", "ci-app-only")
    settings = InfrastructureSettings()
    assert settings.postgres is not None
    assert settings.postgres.password.get_secret_value() == "ci-admin-only"
    assert "ci-admin-only" not in repr(settings)
    assert "ci-app-only" not in repr(settings)


@pytest.mark.parametrize(
    "changes",
    [{"port": 0}, {"port": 65536}, {"host": "production.example"}, {"password": ""}],
)
def test_external_postgres_rejects_invalid_configuration(changes: dict[str, object]) -> None:
    postgres = {"port": 5432, "password": "ci-only", "app_password": "ci-only", **changes}
    with pytest.raises(ValidationError):
        InfrastructureSettings(postgres=postgres)


def test_partial_external_configuration_never_falls_back_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSIGHTPILOT_TEST_POSTGRES__PORT", "5432")
    with pytest.raises(ValidationError):
        InfrastructureSettings()


def test_external_fixture_does_not_start_stop_or_inspect_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = InfrastructureSettings(
        postgres={"port": 5432, "password": "ci-only-admin", "app_password": "ci-only-app"}
    )
    monkeypatch.setattr("tests.shared_database.InfrastructureSettings", lambda: settings)
    docker = Mock(side_effect=AssertionError("External service lifecycle must not use Docker"))
    monkeypatch.setattr("tests.shared_database.require_docker", docker)
    temporary = Mock(side_effect=AssertionError("External service needs no lifecycle artifacts"))
    fixture = pg_container.__wrapped__(temporary)
    postgres = next(fixture)
    assert postgres.volume is None
    assert postgres.app.app_password.get_secret_value() == "ci-only-app"
    assert "ci-only-admin" not in repr(postgres.app)
    with pytest.raises(StopIteration):
        next(fixture)
    docker.assert_not_called()
    temporary.assert_not_called()


@pytest.mark.parametrize("failure", ["cli", "daemon", "timeout", "oserror"])
def test_ci_missing_docker_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("INSIGHTPILOT_TEST_REQUIRE_DOCKER", "true")
    monkeypatch.setattr(
        "tests.shared_database.shutil.which", lambda _: None if failure == "cli" else "/docker"
    )
    error = {
        "timeout": subprocess.TimeoutExpired("docker", 10),
        "oserror": OSError("not available"),
    }.get(failure)
    monkeypatch.setattr(
        "tests.shared_database.subprocess.run",
        Mock(return_value=Mock(returncode=1), side_effect=error),
    )
    with pytest.raises(pytest.fail.Exception, match=r"Docker .* unavailable"):
        require_docker()
