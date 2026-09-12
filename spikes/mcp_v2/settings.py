"""Process-local settings for the MCP spike."""

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENDPOINT = "http://127.0.0.1:9100/mcp"


class MCPProbeSettings(BaseModel):
    """Required runtime token and bounded operation timeout."""

    auth_token: SecretStr = Field(min_length=16)
    timeout_seconds: float = Field(default=10, ge=1, le=60)


class Settings(BaseSettings):
    """Read only dedicated environment variables, never a discovered dotenv."""

    model_config = SettingsConfigDict(
        env_prefix="IP_SPIKE_", env_nested_delimiter="__", extra="forbid", hide_input_in_errors=True
    )
    mcp: MCPProbeSettings
