"""Opt-in CI infrastructure settings, separate from every application process."""

from typing import Annotated, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.settings_base import ConfigModel


class ExternalPostgres(ConfigModel):
    """A runner-local service bootstrapped by CI, never managed by test teardown."""

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    password: Annotated[SecretStr, Field(min_length=1)]
    app_password: Annotated[SecretStr, Field(min_length=1)]


class InfrastructureSettings(BaseSettings):
    """Use a dedicated prefix so runtime process settings cannot consume CI secrets."""

    model_config = SettingsConfigDict(
        env_prefix="INSIGHTPILOT_TEST_",
        env_nested_delimiter="__",
        extra="forbid",
        hide_input_in_errors=True,
    )
    require_docker: bool = False
    postgres: ExternalPostgres | None = None
