"""LLM startup fails closed and HTTP resources close on every lifecycle path."""

from pathlib import Path

import httpx
import pytest

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import LlmConfigurationError
from app.services.health import HealthService
from app.services.llm.service import LlmService
from tests.fakes.health_probe import FakeProbe


async def test_llm_client_owned_by_application(settings: Settings) -> None:
    client = httpx.AsyncClient()
    service = LlmService(settings.llm, client=client)
    app = create_app(
        settings,
        llm_service=service,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
    )
    async with app.router.lifespan_context(app):
        assert app.state.llm is service
        assert not client.is_closed
    assert client.is_closed


async def test_invalid_capabilities_fail_startup_and_close_client(
    settings: Settings, tmp_path: Path
) -> None:
    settings.llm.capabilities_paths = [tmp_path / "absent.json"]
    client = httpx.AsyncClient()
    service = LlmService(settings.llm, client=client)
    app = create_app(
        settings,
        llm_service=service,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
    )
    with pytest.raises(LlmConfigurationError):
        async with app.router.lifespan_context(app):
            pytest.fail("invalid evidence must prevent startup")
    assert client.is_closed
    assert not app.state.ready
