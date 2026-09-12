"""Typed SSH deployment inputs, independent of API credentials and startup."""

from pathlib import Path

from pydantic import Field, FilePath

from app.core.settings_base import ConfigModel, ProcessSettings


class TunnelSettings(ConfigModel):
    """Explicit SSH inputs; no network discovery or key generation happens here."""

    jump_host: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:@-]*$")
    jump_user: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:@-]*$")
    jump_port: int = Field(default=22, ge=1, le=65535)
    target_host: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:@-]*$")
    target_user: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.:@-]*$")
    target_port: int = Field(default=22, ge=1, le=65535)
    jump_key_path: FilePath
    target_key_path: FilePath
    known_hosts_path: FilePath
    config_path: Path
    host_port: int = Field(default=18100, ge=1024, le=65535)


class TunnelProcessSettings(ProcessSettings):
    """Settings for the tunnel wrapper only; runtime service starts in Phase 3."""

    process_name = "tunnel"
    tunnel: TunnelSettings
