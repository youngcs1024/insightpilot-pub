"""Real four-service Compose acceptance; all resources belong to a unique project."""

import json
import secrets
import socket

import pytest
from pydantic import BaseModel, SecretStr

from app.core.config_models import DataAgentSettings, SanitySettings
from app.core.logging import SecretRedactor
from scripts.deployment import (
    Command,
    DeploymentError,
    DeploymentSettings,
    Invocation,
    execute,
)
from scripts.deployment_contracts import environment_issues
from tests.database_support import DatabaseStack

pytestmark = pytest.mark.integration


class CoreStack(BaseModel):
    """Keep operator credentials in the fixture, never in the API query subprocess."""

    docker: str
    settings: DeploymentSettings
    call: Invocation

    def exec_api(self, code: str) -> str:
        """Run fixed validation code with only the container's actual environment."""
        return execute(
            self.docker,
            self.settings,
            self.call.model_copy(
                update={
                    "command": Command.EXEC,
                    "arguments": ["-T", "api", "python", "-c", code],
                }
            ),
        )


@pytest.fixture(scope="module")
def core_stack(
    database_stack: DatabaseStack, tmp_path_factory: pytest.TempPathFactory
) -> CoreStack:
    """Extend this test run's own DB project; its session fixture owns final cleanup.

    Reusing the isolated project avoids exhausting Docker's default subnet pool
    when retained historical projects coexist with both DB test fixtures.
    """
    evidence = tmp_path_factory.mktemp("mcp-container-evidence")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = database_stack.settings.model_copy(
        update={
            "api_host_port": port,
            "data_agent": DataAgentSettings(
                sanity=SanitySettings(
                    money_columns=["gmv"], nonzero_columns=["count"], expected_max_rows=7
                )
            ),
            "mcp_auth_token": SecretStr(secrets.token_urlsafe(32)),
            "jwt_secret": SecretStr(secrets.token_urlsafe(48)),
            "llm_api_key": SecretStr("unused-step17-no-llm-calls"),
        }
    )
    call = database_stack.call.model_copy(
        update={
            "profiles": ["core"],
            "command": Command.UP,
            "arguments": ["-d", "--wait"],
        }
    )
    stack = CoreStack(docker=database_stack.docker, settings=settings, call=call)
    try:
        execute(stack.docker, settings, call)
    except DeploymentError as exc:
        logs = execute(
            stack.docker,
            settings,
            call.model_copy(update={"command": Command.LOGS, "arguments": ["--no-color"]}),
        )
        (evidence / "failure.log").write_text(
            str(SecretRedactor(settings).clean(str(exc.context.get("diagnostics", "")) + logs))
        )
        raise
    return stack


def test_runtime_api_environment_has_only_allowlisted_keys(core_stack: CoreStack) -> None:
    output = core_stack.exec_api(
        "import os,json; from scripts.deployment_contracts import forbidden_api_keys; "
        "keys=set(os.environ); "
        "print(json.dumps(sorted({k for k in keys if k.startswith('IP_')} | set(forbidden_api_keys(keys)))))"
    )
    assert environment_issues("api", set(json.loads(output))).passed


def test_api_container_queries_through_its_lifespan_client(core_stack: CoreStack) -> None:
    # Uses the exact application lifecycle and injected client consumed by later graph nodes.
    output = core_stack.exec_api("""
import asyncio, time
from app.application import create_app
from app.core.config import settings
from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments
async def main():
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        result = await app.state.mcp.call_tool(
            'execute_readonly_query', QueryArguments(sql='SELECT 42 AS answer'),
            deadline=Deadline(time.monotonic()+20))
        assert result.rows == [[42]]
        assert result.columns[0].name == 'answer'
        print('api_mcp_query_verified')
asyncio.run(main())
""")
    assert "api_mcp_query_verified" in output


def test_core_health_and_migration_completed(core_stack: CoreStack) -> None:
    output = core_stack.exec_api("""
import json,urllib.request
for endpoint in ('health','ready'):
    with urllib.request.urlopen('http://127.0.0.1:8000/'+endpoint,timeout=5) as response:
        assert response.status == 200
print('core_ready_verified')
""")
    assert "core_ready_verified" in output
    # Successful service startup also proves service_completed_successfully migration gating.
    rendered = json.loads(
        execute(
            core_stack.docker,
            core_stack.settings,
            core_stack.call.model_copy(
                update={"command": Command.CONFIG, "arguments": ["--format", "json"]},
            ),
        )
    )
    assert (
        rendered["services"]["mcp"]["depends_on"]["migrate"]["condition"]
        == "service_completed_successfully"
    )
    assert not rendered["services"]["mcp"].get("ports")


def test_runtime_data_agent_configuration_is_typed_and_forwarded(core_stack: CoreStack) -> None:
    output = core_stack.exec_api("""
from app.core.config import settings
from app.core.config_models import DataAgentSettings
assert isinstance(settings.data_agent, DataAgentSettings)
print(settings.data_agent.model_dump_json())
""")
    assert DataAgentSettings.model_validate_json(output) == core_stack.settings.data_agent
