"""MCP-only process settings; API and administrator credentials are invalid."""

from typing import Literal

from pydantic import Field

from app.core.settings_base import ConfigModel, ProcessSettings, Secret


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


class ServerSettings(ConfigModel):
    """Internal HTTP listener and bearer secret."""

    auth_token: Secret = Field(repr=False)
    host: str = Field(default="0.0.0.0", min_length=1)  # noqa: S104 -- private container network.
    port: int = Field(default=8001, ge=1, le=65535)
    shutdown_timeout_s: float = Field(default=5, ge=0.01, le=30)


class McpServerSettings(ProcessSettings):
    """Load .env.mcp, never the API's or operator's dotenv file."""

    process_name = "mcp"
    business: BusinessSettings
    mcp: ServerSettings
