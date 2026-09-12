"""API configuration entrypoint: fail immediately when required settings are absent."""

from app.core.config_models import (
    DatabaseSettings,
    HealthSettings,
    HTTPSettings,
    LLMSettings,
    MCPSettings,
    ModelRuntimeClientSettings,
    ObservabilitySettings,
    SchemaCatalogSettings,
    Settings,
)
from app.core.settings_base import Environment
from app.retrieval.config import RetrievalSettings

__all__ = [
    "DatabaseSettings",
    "Environment",
    "HTTPSettings",
    "HealthSettings",
    "LLMSettings",
    "MCPSettings",
    "ModelRuntimeClientSettings",
    "ObservabilitySettings",
    "RetrievalSettings",
    "SchemaCatalogSettings",
    "Settings",
    "settings",
]

settings = Settings.load()
