"""Application cache identity and server metadata coherence through get_schema."""

import pytest

from app.core.config_models import Settings
from app.core.errors import McpUnavailableError, SchemaDriftError
from app.db.session import Database
from app.services.schema_catalog import SchemaCatalogService
from tests.auth_support import deadline
from tests.factories import business_schema
from tests.fakes.mcp_client import FakeMcpClient

pytestmark = pytest.mark.integration


async def test_business_revision_is_propagated_across_cache_refresh(
    auth_database: Database, settings: Settings
) -> None:
    initial = business_schema()
    changed = initial.model_copy(update={"business_revision": "business-v2"}, deep=True)
    client = FakeMcpClient([], schema_responses=[initial, changed, changed])
    catalog = SchemaCatalogService(auth_database, client, settings.schema_catalog, clock=lambda: 0)
    assert await catalog.render(deadline=deadline()) == initial.rendered
    assert await catalog.render(deadline=deadline()) == changed.rendered
    assert catalog._cache.response.business_revision == "business-v2"
    assert (await catalog.validate_against_live_schema(deadline=deadline())).valid
    assert [call.refresh for call in client.schema_calls] == [False, False, True]
    assert not client.calls


@pytest.mark.parametrize("failure", ["content", "unavailable"])
async def test_metadata_mismatch_or_failure_clears_application_cache(
    auth_database: Database, settings: Settings, failure: str
) -> None:
    initial = business_schema()
    next_response = (
        initial.model_copy(update={"metadata_revision": "a" * 64})
        if failure == "content"
        else McpUnavailableError()
    )
    client = FakeMcpClient([], schema_responses=[initial, next_response])
    catalog = SchemaCatalogService(auth_database, client, settings.schema_catalog)
    await catalog.snapshot(deadline=deadline())
    with pytest.raises(SchemaDriftError if failure == "content" else McpUnavailableError):
        await catalog.snapshot(deadline=deadline())
    assert catalog._cache is None
