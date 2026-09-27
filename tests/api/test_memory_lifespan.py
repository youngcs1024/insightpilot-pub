"""Missing uniqueness enforcement prevents readiness, rather than silent legacy operation."""

from unittest.mock import AsyncMock

import pytest

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import MemorySchemaError
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe


async def test_memory_schema_drift_prevents_ready(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    check = AsyncMock(side_effect=MemorySchemaError())
    monkeypatch.setattr("app.application.validate_memory_schema", check)
    app = create_app(settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health))
    with pytest.raises(MemorySchemaError):
        async with app.router.lifespan_context(app):
            pytest.fail("Legacy memory constraints must prevent readiness")
    check.assert_awaited_once()
    assert not app.state.ready
