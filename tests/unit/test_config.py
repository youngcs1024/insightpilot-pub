"""Configuration behavior and import-time acceptance without live service credentials."""

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy.engine import make_url

from app.core.config_models import Settings
from app.core.settings_base import (
    PROJECT_ROOT,
    Environment,
    ProcessSettings,
    environment_keys,
    resolve_process_env_file,
)
from mcp_server.config import McpServerSettings
from model_runtime.config import ModelRuntimeSettings
from model_tunnel.config import TunnelProcessSettings
from scripts.deployment import DeploymentSettings
from scripts.migration_settings import MigrationSettings
from scripts.model_deployment_settings import ModelDeploymentSettings

HOST_DATABASE_PORT = 15432
DEFAULT_POOL_SIZE = 10
ENV_POOL_SIZE = 12
CONSTRUCTOR_POOL_SIZE = 13
EMBED_BATCH = 16
SAMPLE_VALUE = "config-test-credential@:/+ space"
REQUIRED_ENV = {
    "IP_SECURITY__JWT_SECRET": "test-signing-key-only-32-characters-long",
    "IP_DATABASE__APP_PASSWORD": SAMPLE_VALUE,
    "IP_LLM__BASE_URL": "https://provider.example.invalid/v1",
    "IP_LLM__MODEL": "test-model",
    "IP_LLM__API_KEY": SAMPLE_VALUE,
    "IP_MCP__AUTH_TOKEN": SAMPLE_VALUE,
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read developer process secrets or allow test values to escape a test."""
    for key in os.environ:
        if key.upper().startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(ProcessSettings, "project_root", tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("key", REQUIRED_ENV)
def test_missing_required_secret_raises(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(key)
    with pytest.raises(ValidationError) as error:
        Settings.load()
    assert key.removeprefix("IP_").lower().replace("__", ".") in str(error.value)


@pytest.mark.parametrize("size", ["0", "51"])
def test_out_of_range_pool_size_raises(size: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_DATABASE__POOL_SIZE", size)
    with pytest.raises(ValidationError, match=r"greater_than_equal|less_than_equal"):
        Settings.load()


@pytest.mark.parametrize(
    "key",
    [
        "IP_TYPO",
        "IP_DATABASE__POOL_SZE",
        "IP_DATABASE__POOL_SIZE__TYPO",
        "IP_DATABASE__BIZ_PASSWORD",
        "IP_COMPOSE_PROJECT_NAME",
        "IP_BOOTSTRAP__ETL_PASSWORD",
        "IP_TUNNEL__JUMP_KEY_PATH",
        "IP_MODEL_SERVER__AUTH_TOKEN",
        "ip_typo",
    ],
)
def test_unknown_field_rejected(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(key, SAMPLE_VALUE)
    with pytest.raises(ValidationError, match="extra_forbidden") as error:
        Settings.load()
    assert SAMPLE_VALUE not in str(error.value)


@pytest.mark.parametrize("value", ["IP_TYPO=bad", "IP_DATABASE__POOL_SZE=bad", "FOREIGN_KEY=bad"])
def test_dotenv_unknown_fields_rejected(value: str, tmp_path: Path) -> None:
    path = tmp_path / ".env.api"
    path.write_text(value)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings.load()


def test_json_unknown_nested_field_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_DATABASE", '{"pool_sze": 10}')
    with pytest.raises(ValidationError, match="pool_sze"):
        Settings.load()


def test_secret_str_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_RUNTIME__AUTH_TOKEN", SAMPLE_VALUE)
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_PUBLIC_KEY", SAMPLE_VALUE)
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_SECRET_KEY", SAMPLE_VALUE)
    settings = Settings.load()
    assert "password" not in repr(settings)
    assert SAMPLE_VALUE not in repr(settings)
    assert SAMPLE_VALUE not in str(settings)
    assert SAMPLE_VALUE not in settings.model_dump_json()
    assert settings.database.app_password.get_secret_value() == SAMPLE_VALUE


def test_url_quotes_special_characters() -> None:
    url = Settings.load().database.app_url
    parsed = urlsplit(url)
    assert parsed.scheme == "postgresql+asyncpg"
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port == HOST_DATABASE_PORT
    assert parsed.path == "/insightpilot_app"
    assert unquote_plus(parsed.password or "") == SAMPLE_VALUE
    assert "config-test-credential%40%3A%2F%2B%20space" in url
    assert make_url(url).password == SAMPLE_VALUE


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (Settings, ".env.example"),
        (McpServerSettings, ".env.mcp.example"),
        (MigrationSettings, ".env.migration.example"),
        (Settings, ".env.api.container.example"),
        (DeploymentSettings, ".env.deployment.example"),
        (TunnelProcessSettings, ".env.tunnel.example"),
        (ModelRuntimeSettings, ".env.model-runtime.example"),
        (ModelDeploymentSettings, ".env.model-deployment.example"),
    ],
)
def test_env_example_covers_all_required_fields(model: type[BaseModel], name: str) -> None:
    # Also cover optional/default leaves; uncommenting a documented block must stay valid.
    text = (PROJECT_ROOT / name).read_text()
    documented = set(re.findall(r"^\s*(?:#\s*)?(IP_[A-Z0-9_]+)=", text, re.MULTILINE))
    accepted = environment_keys(model)
    leaves = {key for key in accepted if not any(v.startswith(key + "__") for v in accepted)}
    assert documented == leaves


@pytest.mark.parametrize("environment", list(Environment))
def test_environment_defaults(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_ENVIRONMENT", environment.value)
    settings = Settings.load()
    development = environment is Environment.DEVELOPMENT
    assert settings.observability.log_level == ("DEBUG" if development else "INFO")
    assert settings.observability.log_format == ("console" if development else "json")


def test_explicit_defaults_are_not_overridden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IP_ENVIRONMENT", "production")
    monkeypatch.setenv("IP_OBSERVABILITY__LOG_LEVEL", "ERROR")
    (tmp_path / ".env.api.production").write_text("IP_OBSERVABILITY__LOG_FORMAT=console\n")
    settings = Settings.load()
    assert settings.observability.log_level == "ERROR"
    assert settings.observability.log_format == "console"
    explicit = Settings(observability={"log_level": "WARNING"})
    assert explicit.observability.log_level == "WARNING"
    assert explicit.observability.log_format == "console"


@pytest.mark.parametrize("environment", ["prod", "unknown", "", SAMPLE_VALUE])
def test_invalid_environment_rejected(environment: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_ENVIRONMENT", environment)
    with pytest.raises(ValidationError) as error:
        Settings.load()
    assert SAMPLE_VALUE not in str(error.value)


def test_first_existing_file_not_merged(tmp_path: Path) -> None:
    names = [".env.api", ".env.api.local", ".env.api.development", ".env.api.development.local"]
    for index, name in enumerate(names):
        (tmp_path / name).write_text(f"IP_DATABASE__POOL_SIZE={index + 1}\n")
        assert resolve_process_env_file(tmp_path, "api", Environment.DEVELOPMENT) == tmp_path / name
        assert Settings.load().database.pool_size == index + 1
    # The highest file is incomplete: lower files are deliberately not merged into it.
    (tmp_path / names[-1]).write_text("IP_DATABASE__POOL_RECYCLE_S=3600\n")
    assert Settings.load().database.pool_size == DEFAULT_POOL_SIZE


def test_constructor_environment_selects_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IP_ENVIRONMENT", "development")
    (tmp_path / ".env.api.production").write_text("IP_DATABASE__POOL_SIZE=12\n")
    settings = Settings(environment=Environment.PRODUCTION)
    assert settings.database.pool_size == ENV_POOL_SIZE
    assert settings.environment is Environment.PRODUCTION


def test_env_and_constructor_override_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env.api").write_text("IP_DATABASE__POOL_SIZE=11\n")
    monkeypatch.setenv("IP_DATABASE__POOL_SIZE", "12")
    assert Settings.load().database.pool_size == ENV_POOL_SIZE
    assert Settings(database={"pool_size": 13}).database.pool_size == CONSTRUCTOR_POOL_SIZE


def test_explicit_disable_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env.api").write_text("IP_DATABASE__POOL_SIZE=11\n")
    assert Settings(_env_file=None).database.pool_size == DEFAULT_POOL_SIZE


def test_no_foreign_file_or_unprefixed_shell_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    for name in [".env", ".env.api", ".env.deployment"]:
        (foreign / name).write_text("IP_DATABASE__POOL_SIZE=49\n")
    monkeypatch.chdir(foreign)
    monkeypatch.setenv("DATABASE_URL", SAMPLE_VALUE)
    monkeypatch.setenv("APP_ENV", "production")
    settings = Settings.load()
    assert settings.database.pool_size == DEFAULT_POOL_SIZE
    assert settings.environment is Environment.DEVELOPMENT


@pytest.mark.parametrize("key", ["IP_RETRIEVAL__ENABLED", "IP_SEMANTIC_MEMORY_ENABLED"])
def test_model_client_required_only_when_enabled(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings.load().model_runtime is None
    monkeypatch.setenv(key, "true")
    with pytest.raises(ValidationError, match="model_runtime"):
        Settings.load()
    monkeypatch.setenv("IP_MODEL_RUNTIME__AUTH_TOKEN", SAMPLE_VALUE)
    model = Settings.load().model_runtime
    assert model is not None
    assert model.base_url.host == "model-tunnel"
    assert model.embed_batch == EMBED_BATCH
    assert model.max_concurrency == 1


def test_partial_model_client_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MODEL_RUNTIME__BASE_URL", "http://model-tunnel:8100")
    with pytest.raises(ValidationError, match="auth_token"):
        Settings.load()


def test_langfuse_requires_complete_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_ENABLED", "true")
    with pytest.raises(ValidationError, match="enabled Langfuse requires"):
        Settings.load()
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_BASE_URL", "https://trace.example.invalid")
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_PUBLIC_KEY", SAMPLE_VALUE)
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_SECRET_KEY", SAMPLE_VALUE)
    assert Settings.load().observability.langfuse_enabled


@pytest.mark.parametrize("environment", ["development", "test", "staging", "production"])
def test_masking_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch, environment: str) -> None:
    monkeypatch.setenv("IP_ENVIRONMENT", environment)
    monkeypatch.setenv("IP_OBSERVABILITY__LANGFUSE_MASK_DISABLED", "true")
    with pytest.raises(ValidationError, match="trace masking"):
        Settings.load()


@pytest.mark.parametrize(
    "key", ["IP_DATABASE__APP_PASSWORD", "IP_LLM__API_KEY", "IP_MCP__AUTH_TOKEN"]
)
def test_empty_secret_rejected(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(key, "")
    with pytest.raises(ValidationError):
        Settings.load()


def test_validation_messages_hide_raw_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_DATABASE__POOL_SIZE", SAMPLE_VALUE)
    with pytest.raises(ValidationError) as error:
        Settings.load()
    assert SAMPLE_VALUE not in str(error.value)
    assert "input_value" not in str(error.value)


def run_import(
    tmp_path: Path, *, missing_password: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the documented import in an isolated process, using only dummy credentials."""
    env = dict(os.environ)
    env.update(REQUIRED_ENV)
    if missing_password:
        env.pop("IP_DATABASE__APP_PASSWORD")
    # Anchor to an empty isolated project configuration directory before entrypoint import.
    code = (
        "from pathlib import Path; "
        "from app.core.settings_base import ProcessSettings; "
        "import sys; ProcessSettings.project_root = Path(sys.argv[1]); "
        "from app.core.config import settings; print(settings.environment)"
    )
    return subprocess.run(  # noqa: S603 -- fixed interpreter/code and isolated dummy values.
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_import_singleton_succeeds(tmp_path: Path) -> None:
    result = run_import(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "development"


def test_import_singleton_missing_password_fails(tmp_path: Path) -> None:
    result = run_import(tmp_path, missing_password=True)
    assert result.returncode != 0
    assert "database.app_password" in result.stderr
    assert SAMPLE_VALUE not in result.stderr


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MAX_OVERFLOW", "-1"),
        ("MAX_OVERFLOW", "51"),
        ("POOL_TIMEOUT_S", "0"),
        ("POOL_TIMEOUT_S", "121"),
        ("CONNECT_TIMEOUT_S", "0"),
        ("CONNECT_TIMEOUT_S", "61"),
        ("COMMAND_TIMEOUT_S", "0"),
        ("COMMAND_TIMEOUT_S", "121"),
        ("ECHO_SQL", "not-a-boolean"),
    ],
)
def test_database_budget_bounds(key: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_DATABASE__" + key, value)
    with pytest.raises(ValidationError):
        Settings.load()
