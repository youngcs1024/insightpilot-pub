"""Graph resources participate in fail-closed startup and bounded shutdown."""

from unittest.mock import AsyncMock

import pytest

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import CheckpointError
from app.services.graph import GraphService
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe


async def test_graph_owned_by_application(settings: Settings) -> None:
    graph = AsyncMock(spec=GraphService)
    app = create_app(
        settings,
        graph_service=graph,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
    )
    async with app.router.lifespan_context(app):
        assert app.state.graph is graph
        graph.start.assert_awaited_once()
    graph.aclose.assert_awaited_once()


async def test_checkpoint_failure_prevents_startup(settings: Settings) -> None:
    graph = AsyncMock(spec=GraphService)
    graph.start.side_effect = CheckpointError()
    app = create_app(
        settings,
        graph_service=graph,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
    )
    with pytest.raises(CheckpointError):
        async with app.router.lifespan_context(app):
            pytest.fail("checkpoint failure must prevent readiness")
    assert not app.state.ready
    graph.aclose.assert_awaited_once()
