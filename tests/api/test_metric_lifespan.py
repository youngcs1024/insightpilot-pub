"""Metric publication is validated before application readiness."""

from unittest.mock import AsyncMock

import pytest

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import MetricCatalogError
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe


async def test_invalid_metric_catalog_prevents_startup(settings: Settings) -> None:
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    app.state.metrics.validate_startup = AsyncMock(side_effect=MetricCatalogError())
    with pytest.raises(MetricCatalogError):
        async with app.router.lifespan_context(app):
            pytest.fail("Invalid metric catalog must prevent startup")
    assert not app.state.ready
    app.state.metrics.validate_startup.assert_awaited_once()


async def test_metric_catalog_checked_before_ready(settings: Settings) -> None:
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    async with app.router.lifespan_context(app):
        app.state.metrics.validate_startup.assert_awaited_once()
        assert app.state.ready
