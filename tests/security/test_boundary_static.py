"""API-side credential and runtime fallback contracts for the MCP boundary."""

import ast
from pathlib import Path
from typing import get_args, get_origin
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from app.agents.contracts import Route
from app.core.config_models import DatabaseSettings, Settings
from app.core.errors import McpUnavailableError
from tests.agents.parent_support import parent_context
from tests.agents.support import invoke

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
FORBIDDEN_FIELD_PARTS = (
    "business",
    "biz",
    "audit",
    "etl",
    "migration",
    "operator",
    "superuser",
)
FORBIDDEN_LITERALS = {
    "INSIGHTPILOT_BUSINESS",
    "BUSINESS_DATABASE_URL",
    "IP_OPERATOR_BIZ_URL",
}
FORBIDDEN_ENV_PREFIXES = (
    "IP_BUSINESS",
    "IP_BOOTSTRAP",
    "IP_MIGRATION",
    "IP_SEED",
    "IP_OPERATOR",
)


def _nested_model(annotation: object) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if get_origin(annotation) is None:
        return []
    return [model for argument in get_args(annotation) for model in _nested_model(argument)]


def test_no_business_url_in_app_settings() -> None:
    """Walk the actual nested Settings fields without constructing a dotenv singleton."""
    pending = [Settings]
    seen: set[type[BaseModel]] = set()
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        for name, field in model.model_fields.items():
            assert not any(part in name.lower() for part in FORBIDDEN_FIELD_PARTS), (model, name)
            pending.extend(_nested_model(field.annotation))
    assert DatabaseSettings.model_fields["app_db"].default == "insightpilot_app"
    assert DatabaseSettings.model_fields["app_user"].default == "app_rw"
    assert "business_url" not in vars(DatabaseSettings)


def test_no_module_imports_business_url() -> None:
    """Reject credential-bearing MCP imports and literal business connection settings."""
    for path in APP_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name == "mcp_server" or alias.name.startswith("mcp_server.")
                    for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module != "mcp_server" and not module.startswith("mcp_server."), path
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.upper()
                assert value not in FORBIDDEN_LITERALS, path
                assert not value.startswith(FORBIDDEN_ENV_PREFIXES), path


async def test_mcp_down_never_connects_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOTH turn keeps knowledge evidence without opening any database fallback."""
    connect = AsyncMock(side_effect=AssertionError("business database fallback"))
    monkeypatch.setattr("app.db.session.create_async_engine", connect)
    monkeypatch.setattr("psycopg.AsyncConnection.connect", connect)
    monkeypatch.setattr("asyncpg.connect", connect)
    output = await invoke(parent_context(Route.BOTH, data_error=McpUnavailableError()))
    assert output.status == "degraded"
    assert output.answer is not None
    assert "数据源当前不可用" in output.answer.markdown
    assert "企业知识库" in output.answer.markdown
    connect.assert_not_called()
