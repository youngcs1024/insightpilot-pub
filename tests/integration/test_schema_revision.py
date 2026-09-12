"""Revision handshake contracts exercise the catalog service without invoking a graph."""

import pytest

from app.core.config_models import Settings
from app.core.errors import McpResultError
from app.db.session import Database
from app.schemas.schema_catalog import BusinessSchemaResponse
from app.services.schema_catalog import SchemaCatalogService
from tests.auth_support import deadline
from tests.factories import business_schema
from tests.fakes.mcp_client import FakeMcpClient

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("cached", [False, True])
async def test_schema_unchanged_requires_matching_cached_revision(
    auth_database: Database, settings: Settings, cached: bool
) -> None:
    client = FakeMcpClient(
        [],
        schema_responses=[
            *([business_schema()] if cached else []),
            BusinessSchemaResponse(
                revision="different" if cached else "business-v1", unchanged=True
            ),
        ],
    )
    catalog = SchemaCatalogService(auth_database, client, settings.schema_catalog, clock=lambda: 0)
    if cached:
        await catalog.snapshot(deadline=deadline())
    with pytest.raises(McpResultError):
        await catalog.render(deadline=deadline())
    assert not client.calls


async def test_schema_revision_is_propagated_across_cache_refresh(
    auth_database: Database, settings: Settings
) -> None:
    initial = business_schema()
    changed = initial.model_copy(update={"revision": "business-v2"}, deep=True)
    client = FakeMcpClient(
        [],
        schema_responses=[
            initial,
            BusinessSchemaResponse(revision=initial.revision, unchanged=True),
            changed,
            BusinessSchemaResponse(revision=changed.revision, unchanged=True),
        ],
    )
    catalog = SchemaCatalogService(auth_database, client, settings.schema_catalog, clock=lambda: 0)
    first = await catalog.render(deadline=deadline())
    assert first
    assert await catalog.render(deadline=deadline()) == first
    updated = await catalog.snapshot(deadline=deadline())
    assert await catalog.snapshot(deadline=deadline()) == updated
    assert [call.known_revision for call in client.schema_calls] == [
        None,
        initial.revision,
        initial.revision,
        changed.revision,
    ]
    assert not client.calls
