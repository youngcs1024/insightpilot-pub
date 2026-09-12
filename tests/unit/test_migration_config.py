"""Configuration and offline migration safety without requiring PostgreSQL."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from pydantic import SecretStr, ValidationError
from sqlalchemy import MetaData, Table, make_url

from alembic import command
from app.core.config_models import Settings
from scripts.migrate_all import migration_config
from scripts.migration_environment import EXCLUDE_TABLES, configure_context, include_object
from scripts.migration_settings import MigrationDatabaseSettings, MigrationSettings, MigrationTarget

if TYPE_CHECKING:
    from sqlalchemy.schema import SchemaItem


@pytest.fixture(autouse=True)
def isolated_migration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never inherit developer process credentials in configuration tests."""
    for key in os.environ:
        if key.startswith("IP_"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("target", list(MigrationTarget))
def test_operator_url_roundtrips_special_characters(target: MigrationTarget) -> None:
    special_characters = "@:/+'$\\\" space%"
    settings = MigrationDatabaseSettings(password=SecretStr(special_characters))
    url = make_url(settings.url(target))
    assert url.password == special_characters
    assert url.database == (
        "insightpilot_app" if target == MigrationTarget.APP else "insightpilot_business"
    )
    assert special_characters not in repr(settings)


def test_migration_credentials_rejected_by_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MIGRATION__PASSWORD", "operator-only")
    with pytest.raises(ValidationError, match="IP_MIGRATION__PASSWORD"):
        Settings(_env_file=None)
    assert "migration" not in Settings.model_fields


def test_migration_config_does_not_accept_runtime_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_MCP__AUTH_TOKEN", "not-for-migrations")
    with pytest.raises(ValidationError, match="IP_MCP__AUTH_TOKEN"):
        MigrationSettings(_env_file=None)


def test_migration_env_example_and_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = Path(".env.migration.example").read_text()
    (tmp_path / ".env.migration.test.local").write_text(example)
    monkeypatch.setattr(MigrationSettings, "project_root", tmp_path)
    monkeypatch.setenv("IP_ENVIRONMENT", "test")
    override_port = 25432
    monkeypatch.setenv("IP_MIGRATION__PORT", str(override_port))
    settings = MigrationSettings.load()
    assert settings.migration.port == override_port
    assert settings.migration.user == "postgres"
    assert "IP_MIGRATION__PASSWORD" in example


@pytest.mark.parametrize("field", ["connect_timeout_s", "command_timeout_s", "lock_timeout_s"])
def test_migration_timeouts_have_bounds(field: str) -> None:
    with pytest.raises(ValidationError):
        MigrationDatabaseSettings.model_validate({"password": "test", field: 0})


@pytest.mark.parametrize("target", list(MigrationTarget))
def test_offline_sql_has_owner_and_independent_version_table(target: MigrationTarget) -> None:
    config = migration_config(target)
    output = io.StringIO()
    config.output_buffer = output
    command.upgrade(config, "head", sql=True)
    sql = output.getvalue()
    assert (
        "SET LOCAL ROLE " + ("app_owner" if target == MigrationTarget.APP else "biz_owner") in sql
    )
    assert (
        "alembic_version_app" if target == MigrationTarget.APP else "biz.alembic_version_biz"
    ) in sql
    assert all("DROP TABLE " + name not in sql for name in EXCLUDE_TABLES)
    if target == MigrationTarget.APP:
        assert "CREATE EXTENSION IF NOT EXISTS citext" in sql
        assert "CREATE TABLE users" in sql
    else:
        assert "CREATE SCHEMA biz" not in sql
        assert "CREATE SCHEMA ops AUTHORIZATION biz_owner" in sql
        assert "CREATE TABLE ops.seed_manifest" in sql
        assert "CREATE TABLE biz.orders" in sql
        assert "CREATE TABLE users" not in sql


@pytest.mark.parametrize("target", list(MigrationTarget))
def test_comparison_options_and_exclusions(target: MigrationTarget) -> None:
    config = migration_config(target)
    with EnvironmentContext(
        config, ScriptDirectory.from_config(config), as_sql=True
    ) as environment:
        configure_context(environment, target, url="postgresql+asyncpg://")
        options = environment.get_context().opts
        assert options["compare_type"] is True
        assert options["compare_server_default"] is True
        assert options["include_object"] is include_object
        assert options["include_schemas"] == (target == MigrationTarget.BUSINESS)
    # Shape is irrelevant to the table-name exclusion, including offline use.
    obj: SchemaItem = Table("test", MetaData())
    for name in EXCLUDE_TABLES:
        assert not include_object(obj, name, "table", True, None)
        assert not include_object(obj, name, "table", False, None)
    assert include_object(obj, "users", "table", True, None)
    assert include_object(obj, "checkpoints", "index", True, None)
