"""Regression tests for the shared fakes, settings and architecture guardrails."""

import time
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.agents.contracts import PhaseOneSqlGeneratorOutput
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, McpUnavailableError
from app.core.llm_config import ModelRole
from app.schemas.mcp import QueryArguments
from tests import factories
from tests.architecture_rules import violations
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.settings_support import SettingsOverride
from tests.shared_database import require_docker

BCRYPT_ROUNDS = 5


def deadline() -> Deadline:
    return Deadline(time.monotonic() + 10)


async def test_fake_llm_langchain_and_structured_contract() -> None:
    value = PhaseOneSqlGeneratorOutput(thinking="count", sql_query="SELECT 42")
    model = FakeChatModel(["sync", "async", value, value.model_dump_json()])
    assert isinstance(model, BaseChatModel)
    assert model.invoke("hello").content == "sync"
    assert (await model.ainvoke("hello")).content == "async"
    for _ in range(2):
        assert (
            await model.generate_structured(ModelRole.SQL, [], type(value), deadline=deadline())
            == value
        )
    with pytest.raises(AssertionError, match="queue exhausted"):
        await model.ainvoke("extra")


async def test_fake_llm_detaches_calls_and_responses() -> None:
    response = PhaseOneSqlGeneratorOutput(thinking="count", sql_query="SELECT 42")
    model = FakeChatModel([response])
    response.sql_query = "SELECT 0"
    prompt = HumanMessage(content="original")
    result = await model.generate_structured(
        ModelRole.SQL, [prompt], type(response), deadline=deadline()
    )
    prompt.content = "mutated"
    model.calls[0].messages[0].content = "also mutated"
    assert model.calls[0].messages[0].content == "original"
    assert model.calls[0].schema_name == "PhaseOneSqlGeneratorOutput"
    assert model.calls[0].role == ModelRole.SQL
    assert result.sql_query == "SELECT 42"


async def test_fake_llm_validation_error_and_deadline() -> None:
    model = FakeChatModel(["{}", McpUnavailableError(), "remaining"])
    with pytest.raises(ValidationError):
        await model.generate_structured(
            ModelRole.SQL, [], PhaseOneSqlGeneratorOutput, deadline=deadline()
        )
    with pytest.raises(McpUnavailableError):
        await model.ainvoke("error")
    with pytest.raises(DeadlineExceededError):
        await model.generate_structured(
            ModelRole.SQL, [], PhaseOneSqlGeneratorOutput, deadline=Deadline(0)
        )
    assert (await model.ainvoke("last")).content == "remaining"


async def test_fake_mcp_records_success_empty_and_failure() -> None:
    model = FakeMcpClient([factories.query_result([]), McpUnavailableError()])
    arguments = QueryArguments(sql="SELECT 42")
    with pytest.raises(DeadlineExceededError):
        await model.call_tool("execute_readonly_query", arguments, deadline=Deadline(0))
    assert not model.calls
    result = await model.call_tool("execute_readonly_query", arguments, deadline=deadline())
    assert result.rows == []
    arguments.sql = "SELECT 0"
    model.calls[0].arguments.sql = "SELECT 1"
    assert model.calls[0].arguments.sql == "SELECT 42"
    with pytest.raises(McpUnavailableError):
        await model.call_tool("execute_readonly_query", arguments, deadline=deadline())
    with pytest.raises(AssertionError, match="queue exhausted"):
        await model.call_tool("execute_readonly_query", arguments, deadline=deadline())


def test_settings_override_validates_and_restores(settings: Settings) -> None:
    original = settings.security.bcrypt_rounds
    with pytest.MonkeyPatch.context() as patch:
        override = SettingsOverride(patch, settings)
        override(settings.security, bcrypt_rounds=BCRYPT_ROUNDS)
        assert settings.security.bcrypt_rounds == BCRYPT_ROUNDS
        with pytest.raises(ValidationError):
            override(settings.security, bcrypt_rounds=-1)
        assert settings.security.bcrypt_rounds == BCRYPT_ROUNDS
    assert settings.security.bcrypt_rounds == original


def test_settings_override_fixture(override_settings: SettingsOverride, settings: Settings) -> None:
    override_settings(security={**settings.security.model_dump(), "bcrypt_rounds": BCRYPT_ROUNDS})
    assert settings.security.bcrypt_rounds == BCRYPT_ROUNDS


def test_factories_are_independent_and_unsaved() -> None:
    first, second = factories.user(), factories.user()
    assert first.id != second.id
    assert first.email != second.email
    conversation = factories.conversation(first.id)
    turn = factories.turn(conversation.id)
    assert conversation.user_id == first.id
    assert turn.conversation_id == conversation.id
    from sqlalchemy import inspect  # noqa: PLC0415 -- test-only inspection.

    assert all(inspect(item).transient for item in (first, second, conversation, turn))


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("import asyncio as a; a.create_task(coro())", "no_bare_create_task"),
        ("from asyncio import create_task as spawn; spawn(coro())", "no_bare_create_task"),
        ("raise HTTPException(500, detail=str(error))", "no_detail_str_e"),
        ("import sqlalchemy as sa; sa.create_engine('url')", "no_sync_engine_in_app"),
        ("from sqlalchemy import create_engine as engine; engine('url')", "no_sync_engine_in_app"),
        ("from app.db import session as s", "nodes_do_no_io"),
        ("from ...clients import mcp_client as m", "nodes_do_no_io"),
        ("import httpx as client", "nodes_do_no_io"),
    ],
)
def test_architecture_rules_detect_aliases(source: str, rule: str) -> None:
    assert violations(source, "app/agents/nodes/example.py")[rule]


def test_architecture_rules_allow_injected_services_and_background_owner() -> None:
    source = "from app.agents.runtime import RuntimeContext\n# asyncio.create_task(coro())\ngroup.create_task(coro())\n"
    assert not any(violations(source, "app/agents/nodes/example.py").values())
    assert not any(
        violations("import asyncio; asyncio.create_task(coro())", "app/core/background.py").values()
    )


def test_missing_docker_is_explicit_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSIGHTPILOT_TEST_REQUIRE_DOCKER", "false")
    monkeypatch.setattr("tests.shared_database.shutil.which", lambda _: None)
    with pytest.raises(pytest.skip.Exception, match="acceptance remains pending"):
        require_docker()


def test_unavailable_daemon_is_explicit_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSIGHTPILOT_TEST_REQUIRE_DOCKER", "false")
    monkeypatch.setattr("tests.shared_database.shutil.which", lambda _: "/docker")
    monkeypatch.setattr(
        "tests.shared_database.subprocess.run", Mock(return_value=Mock(returncode=1))
    )
    with pytest.raises(pytest.skip.Exception, match="daemon unavailable"):
        require_docker()


async def test_fake_fixtures_accept_scripts(
    fake_llm: FakeChatModel, fake_mcp: FakeMcpClient
) -> None:
    fake_llm.enqueue("configured")
    assert (await fake_llm.ainvoke("question")).content == "configured"
    fake_mcp.enqueue(factories.query_result())
    payload = await fake_mcp.call_tool(
        "execute_readonly_query", QueryArguments(sql="SELECT 42"), deadline=deadline()
    )
    assert payload.row_count == 1


def test_meta_allows_redacted_logging_but_rejects_response_detail() -> None:
    source = 'import structlog\nlog = structlog.get_logger()\nlog.exception("failed", detail=str(exc))\nraise HTTPException(500, detail=str(exc))'
    assert violations(source, "app/api/example.py")["no_detail_str_e"] == [4]


def test_migration_failure_is_not_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.migration_environment import MigrationError  # noqa: PLC0415
    from tests.shared_database import TestPostgres, migrated_db  # noqa: PLC0415

    upgrade = Mock(side_effect=MigrationError())
    monkeypatch.setattr("tests.shared_database.command.upgrade", upgrade)
    synthetic = "fixture-only"
    postgres = TestPostgres(port=15432, password=synthetic, app_password=synthetic, volume="unused")
    with pytest.raises(MigrationError):
        migrated_db.__wrapped__(postgres)
    upgrade.assert_called_once()
