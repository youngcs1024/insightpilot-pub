"""Synthetic configuration and validated, automatically restored overrides."""

import os

import pytest
from pydantic import BaseModel

from app.core.config_models import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Construct typed process settings using unmistakably synthetic credentials."""
    for name in os.environ:
        if name.upper().startswith("IP_"):
            monkeypatch.delenv(name)
    return Settings(
        _env_file=None,
        database={"app_password": "health-test-password@:/+"},
        llm={
            "base_url": "https://provider.invalid/v1",
            "model": "qwen3.6-flash-2026-04-16",
            "api_key": "test-llm-key",
        },
        security={"jwt_secret": "test-signing-key-only-32-characters-long", "bcrypt_rounds": 4},
        mcp={"auth_token": "health-test-mcp-token"},
        observability={"log_format": "json"},
        router={"strategy": "hybrid"},  # Preserve recorded pre-Step-4.11 model call sequences.
    )


class SettingsOverride:
    """Revalidate the complete model before monkeypatching any changed field."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
        self.monkeypatch = monkeypatch
        self.settings = settings

    def __call__(self, target: BaseModel | None = None, **changes: object) -> None:
        model = self.settings if target is None else target
        validated = type(model).model_validate({**model.model_dump(), **changes})
        for name in changes:
            self.monkeypatch.setattr(model, name, getattr(validated, name))


@pytest.fixture
def override_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> SettingsOverride:
    """Override a full nested model or fields on one model; restore after each test."""
    return SettingsOverride(monkeypatch, settings)
