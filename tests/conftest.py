"""Shared, explicitly isolated PostgreSQL fixture for real database tests."""

import pytest

from tests import auth_support, database_support, milvus_support, settings_support, shared_api, shared_database
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient

# Dedicated hardware/live modules are excluded before import, including collect-only.
collect_ignore = ["gpu", "live"]
pytest_plugins = ("tests.collection_support", "tests.storage_tracking")



milvus_stack = milvus_support.milvus_stack
database_stack = database_support.database_stack
# API idempotency and migration checks share one schema-owning stack, separate
# from Step 1.2 metadata-only tests. Do not register it once per test module.
migration_stack = database_support.database_stack


auth_database = auth_support.auth_database
unauthenticated_client = auth_support.unauthenticated_client
migrated = auth_support.migrated


settings = settings_support.settings
override_settings = settings_support.override_settings
pg_container = shared_database.pg_container
migrated_db = shared_database.migrated_db
db_connection = shared_database.db_connection
db_session = shared_database.db_session
client = shared_api.client
auth_client = shared_api.auth_client


@pytest.fixture
def fake_llm() -> FakeChatModel:
    """Empty strict queue; construct a scripted instance for each scenario."""
    return FakeChatModel([])


@pytest.fixture
def fake_mcp() -> FakeMcpClient:
    """Empty strict queue prevents accidental unconfigured tool successes."""
    return FakeMcpClient([])
