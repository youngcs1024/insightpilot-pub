"""Typed role configuration shared by settings and the LLM registry."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.core.settings_base import ConfigModel


class ModelRole(StrEnum):
    """Stable role names; every generation chooses a role explicitly."""

    ROUTER = "router"
    SQL = "sql"
    SYNTHESIS = "synthesis"
    MEMORY_EXTRACT = "memory_extract"
    SUMMARIZE = "summarize"
    COMPRESS = "compress"


class ModelRoleSettings(ConfigModel):
    """Missing model and timeout inherit the enclosing LLM settings."""

    model: str | None = Field(default=None, min_length=1, max_length=200)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    timeout_s: float | None = Field(default=None, ge=0.01, le=120)
    fallback_models: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=8
    )
