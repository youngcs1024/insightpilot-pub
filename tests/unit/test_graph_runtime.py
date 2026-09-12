"""Checkpoint lifecycle validation with controlled async pool behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config_models import DatabaseSettings
from app.core.errors import CheckpointError
from app.services.graph import GraphService, serializer
from tests.agents.support import context

TEST_CREDENTIAL = "test-runtime-secret"


@pytest.mark.parametrize("versions", [[], [0], list(range(len(AsyncPostgresSaver.MIGRATIONS) + 1))])
async def test_checkpoint_version_mismatch_closes_pool(
    versions: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = AsyncMock()
    connection = pool.connection.return_value.__aenter__.return_value
    connection.execute.return_value.fetchall.return_value = [{"v": v} for v in versions]
    # connection() is a synchronous context-manager factory, not an async function.

    pool.connection = MagicMock(return_value=pool.connection.return_value)
    monkeypatch.setattr("app.services.graph.AsyncConnectionPool", lambda *args, **kwargs: pool)
    graph = GraphService(DatabaseSettings(app_password=TEST_CREDENTIAL))
    with pytest.raises(CheckpointError):
        await graph.start()
    assert graph.graph is None
    assert graph.pool is None
    pool.close.assert_awaited_once()


def test_serializer_does_not_enable_pickle_fallback() -> None:
    assert serializer().pickle_fallback is False


async def test_unstarted_graph_is_typed_failure() -> None:
    graph = GraphService(DatabaseSettings(app_password=TEST_CREDENTIAL))
    with pytest.raises(CheckpointError):
        await graph.invoke(context())
    await graph.aclose()


async def test_missing_checkpoint_table_is_typed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = AsyncMock()
    manager = pool.connection.return_value
    connection = manager.__aenter__.return_value
    connection.execute.return_value.fetchall.return_value = [
        {"v": v} for v in range(len(AsyncPostgresSaver.MIGRATIONS))
    ]
    connection.execute.return_value.fetchone.return_value = {"relation": None}
    pool.connection = MagicMock(return_value=manager)
    monkeypatch.setattr("app.services.graph.AsyncConnectionPool", lambda *args, **kwargs: pool)
    graph = GraphService(DatabaseSettings(app_password=TEST_CREDENTIAL))
    with pytest.raises(CheckpointError):
        await graph.start()
    pool.close.assert_awaited_once()
