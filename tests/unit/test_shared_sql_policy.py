"""Old server imports and API preflight share exactly one policy implementation."""

import pytest
from langgraph.runtime import Runtime

from app.agents.data.nodes.lifecycle import validate_sql
from app.agents.data.state import DataAgentState
from app.core.sql_policy.allowlist import ALLOWED_TABLES
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import ValidationStatus
from mcp_server.policy.allowlist import ALLOWED_TABLES as SERVER_TABLES
from mcp_server.policy.sql_validator import SQLValidator as ServerValidator
from tests.agents.support import context


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id FROM biz.orders",
        "SELECT 1 UNION SELECT 2 LIMIT 9999",
        "WITH o AS (SELECT order_id FROM biz.orders) SELECT * FROM o",
        "DELETE FROM biz.orders",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT pg_sleep(1)",
        "SELECT 1; SELECT 2",
        "SELECT (",
    ],
)
def test_shared_client_server_policy_equivalence(sql: str) -> None:
    assert SQLValidator is ServerValidator
    assert ALLOWED_TABLES is SERVER_TABLES
    outcome = SQLValidator().validate(sql)
    assert outcome == ServerValidator().validate(sql)
    value = DataAgentState(question="test", generated_sql=sql)
    update = validate_sql(value, Runtime(context=context()))
    if outcome.status is ValidationStatus.VALID:
        assert not update.goto
        assert update.update == {}
    else:
        assert update.goto == "route_correction"
        assert update.update["failures"]
    assert value.generated_sql == sql
