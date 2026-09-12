"""All canonical SQL executes on a migrated baseline through the real MCP boundary."""

import pytest

from app.clients.mcp_client import McpClient
from evals.harness.compare import compare
from evals.harness.contracts import Case, Comparison
from evals.harness.dataset import load_cases
from scripts.seed_settings import SeedSettings
from tests.database_support import DatabaseStack
from tests.integration.catalog_support import catalog_migrated, client
from tests.integration.mcp_support import mcp_endpoint, query
from tests.seed_support import seed_database_stack, seed_directory, seeded

pytestmark = pytest.mark.integration
# Fixture bodies are deferred until test execution; collection only reads static YAML.
database_stack = seed_database_stack
server_endpoint = mcp_endpoint
__all__ = ["catalog_migrated", "client", "seed_directory", "seeded"]


@pytest.fixture(scope="module")
def seed_stack(database_stack: DatabaseStack) -> DatabaseStack:
    return database_stack


CASES = [case for case in load_cases() if not case.adversarial_sql]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_all_canonical_sql_executes(
    case: Case, client: McpClient, seeded: SeedSettings
) -> None:
    result = await query(client, case.canonical_sql)
    assert not result.result_truncated
    if case.comparison is Comparison.EMPTY:
        assert result.row_count == 0
    else:
        assert result.row_count > 0
    if case.anchor_rows is not None:
        expected = result.model_copy(
            update={"rows": case.anchor_rows, "row_count": len(case.anchor_rows)}
        )
        assert compare(result, expected, case.comparison, case.tolerance)
