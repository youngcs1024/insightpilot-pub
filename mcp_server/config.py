"""MCP-only process settings; API and administrator credentials are invalid."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_core import PydanticCustomError

from app.core.settings_base import ConfigModel, ProcessSettings, Secret

MIN_MCP_TOKEN_CHARS = 32


class BusinessSettings(ConfigModel):
    """The sole business database login available to the query executor."""

    host: str = Field(default="postgres", min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    database: Literal["insightpilot_business"] = "insightpilot_business"
    user: Literal["mcp_ro"] = "mcp_ro"
    password: Secret = Field(repr=False)
    pool_size: int = Field(default=10, ge=1, le=50)
    connect_timeout_s: int = Field(default=5, ge=1, le=30)
    operation_timeout_s: float = Field(default=12, ge=0.01, le=30)


class AuditSettings(ConfigModel):
    """Independent write-only database login for MCP audit events."""

    host: str = Field(default="postgres", min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    database: Literal["insightpilot_business"] = "insightpilot_business"
    user: Literal["mcp_audit"] = "mcp_audit"
    password: Secret = Field(repr=False)
    pool_size: int = Field(default=2, ge=1, le=10)
    connect_timeout_s: int = Field(default=2, ge=1, le=10)
    operation_timeout_s: float = Field(default=2, ge=0.01, le=10)


class ServerSettings(ConfigModel):
    """Internal HTTP listener and bearer secret."""

    auth_token: Secret = Field(repr=False)
    host: str = Field(default="0.0.0.0", min_length=1)  # noqa: S104 -- private container network.
    port: int = Field(default=8001, ge=1, le=65535)
    shutdown_timeout_s: float = Field(default=5, ge=0.01, le=30)

    @field_validator("auth_token")
    @classmethod
    def check_auth_token_length(cls, value: SecretStr) -> SecretStr:
        """Reject weak or placeholder credentials before opening server resources."""
        if len(value.get_secret_value()) < MIN_MCP_TOKEN_CHARS:
            raise PydanticCustomError(
                "mcp_token_length", "MCP bearer token must contain at least 32 characters"
            )
        return value


class SchemaSettings(ConfigModel):
    """Baked metadata location and disposable physical-schema cache lifetime."""

    artifact_path: Path = Path(__file__).resolve().parent / "data/schema_metadata.json"
    ttl_s: float = Field(default=300, ge=1, le=3600)


class MetricPolicySettings(ConfigModel):
    """Bound the window accepted for a rendered metric query."""

    max_period_years: int = Field(default=5, ge=1, le=10)


class McpServerSettings(ProcessSettings):
    """Load .env.mcp, never the API's or operator's dotenv file."""

    process_name = "mcp"
    business: BusinessSettings
    audit: AuditSettings
    mcp: ServerSettings
    schema_metadata: SchemaSettings = Field(default_factory=SchemaSettings)
    metric_policy: MetricPolicySettings = Field(default_factory=MetricPolicySettings)
