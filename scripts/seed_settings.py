"""ETL-only settings, never imported or instantiated by the API."""

from typing import ClassVar, Literal

from pydantic import Field

from app.core.config_models import DatabaseSettings
from app.core.settings_base import ConfigModel, ProcessSettings, Secret


class SeedDatabaseSettings(ConfigModel):
    """Only the business ETL login can be selected."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=15432, ge=1, le=65535)
    user: Literal["etl_rw"] = "etl_rw"
    password: Secret = Field(repr=False)
    connect_timeout_s: float = Field(default=5, ge=0.01, le=60)
    command_timeout_s: float = Field(default=60, ge=0.01, le=120)
    lock_timeout_s: float = Field(default=5, ge=0.01, le=60)
    batch_size: int = Field(default=1000, ge=1, le=5000)

    @property
    def url(self) -> str:
        """Use the existing tested credential-safe URL encoding."""
        return DatabaseSettings(
            host=self.host,
            port=self.port,
            app_user=self.user,
            app_password=self.password,
            app_db="insightpilot_business",
        ).app_url


class SeedSettings(ProcessSettings):
    """Read only seed process environment and local .env.seed configuration."""

    process_name: ClassVar[str] = "seed"
    seed: SeedDatabaseSettings
