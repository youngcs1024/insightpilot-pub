"""Operator-only configuration; never imported by the API process."""

from enum import StrEnum
from typing import ClassVar

from pydantic import Field

from app.core.config_models import DatabaseSettings
from app.core.settings_base import ConfigModel, ProcessSettings, Secret


class MigrationTarget(StrEnum):
    """Independent migration histories, not branches in a shared history."""

    APP = "app"
    BUSINESS = "business"


class MigrationDatabaseSettings(ConfigModel):
    """Administrative connection with bounded connection and statement waits."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=15432, ge=1, le=65535)
    user: str = Field(default="postgres", min_length=1)
    password: Secret = Field(repr=False)
    connect_timeout_s: float = Field(default=5, ge=0.01, le=60)
    command_timeout_s: float = Field(default=30, ge=0.01, le=120)
    lock_timeout_s: float = Field(default=5, ge=0.01, le=60)

    def url(self, target: MigrationTarget) -> str:
        """Reuse the tested asyncpg URL encoding, including spaces and reserved characters."""
        return DatabaseSettings(
            host=self.host,
            port=self.port,
            app_user=self.user,
            app_password=self.password,
            app_db="insightpilot_app" if target == MigrationTarget.APP else "insightpilot_business",
        ).app_url


class MigrationSettings(ProcessSettings):
    """Load only project-local .env.migration files and migration-scoped fields."""

    process_name: ClassVar[str] = "migration"
    migration: MigrationDatabaseSettings
