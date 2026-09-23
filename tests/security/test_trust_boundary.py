"""Prove the boundary with real PostgreSQL errors and owner-created table privileges."""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote_plus

import asyncpg
import psycopg
import pytest
from psycopg import sql
from pydantic import SecretStr

from app.core.errors import SqlExecutionError
from mcp_server.tools.execute_query import QueryExecutor
from scripts.deployment import Command, execute
from scripts.deployment_contracts import environment_issues
from tests.integration.mcp_support import business, business_tables

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp_server.db import BusinessDatabase
    from tests.database_support import DatabaseStack

pytestmark = pytest.mark.integration
__all__ = ["business", "business_tables"]
Role = Literal["postgres", "app_rw", "etl_rw", "mcp_ro", "mcp_audit"]
Database = Literal["insightpilot_app", "insightpilot_business"]


@asynccontextmanager
async def connection(
    stack: DatabaseStack, role: Role, database: Database = "insightpilot_business"
) -> AsyncIterator[asyncpg.Connection]:
    """Authenticate over TCP, with explicit connect and query deadlines."""
    settings = stack.settings
    secret = {
        "postgres": settings.postgres_superuser_password,
        "app_rw": settings.bootstrap.app_password,
        "etl_rw": settings.bootstrap.etl_password,
        "mcp_ro": settings.bootstrap.mcp_password,
        "mcp_audit": settings.bootstrap.audit_password,
    }[role]
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=settings.db_host_port,
        user=role,
        password=secret.get_secret_value(),
        database=database,
        timeout=5,
        command_timeout=20,
    )
    try:
        yield conn
    finally:
        await conn.close(timeout=5)


async def test_app_role_cannot_connect_to_business_db(database_stack: DatabaseStack) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with connection(database_stack, "app_rw"):
            pytest.fail("app_rw connected to the business database")


async def test_audit_role_is_confined_to_mcp_schema(database_stack: DatabaseStack) -> None:
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        async with connection(database_stack, "mcp_audit", "insightpilot_app"):
            pytest.fail("mcp_audit connected to the application database")
    async with connection(database_stack, "mcp_audit") as conn:
        assert await conn.fetchval("SELECT has_schema_privilege(current_user, 'mcp', 'USAGE')")
        assert not await conn.fetchval("SELECT has_schema_privilege(current_user, 'biz', 'USAGE')")
        assert not await conn.fetchval("SELECT has_schema_privilege(current_user, 'mcp', 'CREATE')")


async def test_mcp_role_can_select_from_biz(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "mcp_ro") as conn:
        assert await conn.fetchval("SELECT 1") == 1
        assert await conn.fetchval("SELECT has_schema_privilege(current_user, 'biz', 'USAGE')")


async def test_mcp_role_cannot_insert(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "mcp_ro") as conn:
        with pytest.raises(
            (asyncpg.ReadOnlySQLTransactionError, asyncpg.InsufficientPrivilegeError)
        ):
            await conn.execute("CREATE TABLE biz.t(i int)")


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO biz.regions(region_id) VALUES (99)",
        "UPDATE biz.regions SET region_id = 99",
        "DELETE FROM biz.regions",
        "CREATE TABLE biz.boundary_forbidden(value integer)",
    ],
)
async def test_mcp_role_cannot_insert_update_delete_create(
    database_stack: DatabaseStack, business_tables: None, statement: str
) -> None:
    """The login's read-only default rejects all four write shapes."""
    async with connection(database_stack, "mcp_ro") as conn:
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await conn.execute(statement)


async def test_mcp_role_read_only_transaction_enforced(database_stack: DatabaseStack) -> None:
    """A new transaction inherits the role restriction before any MCP policy runs."""
    async with connection(database_stack, "mcp_ro") as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            assert await conn.fetchval("SHOW transaction_read_only") == "on"
            with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
                await conn.execute("CREATE TABLE biz.boundary_forbidden(value integer)")
        finally:
            await transaction.rollback()


async def test_executor_called_directly_still_rejected(business: BusinessDatabase) -> None:
    """Bypass SQLValidator: the actual executor and database still reject INSERT."""
    executor = QueryExecutor(business)
    before = await executor.execute("SELECT count(*) AS n FROM biz.regions WHERE region_id = 99")
    with pytest.raises(SqlExecutionError) as caught:
        await executor.execute("INSERT INTO biz.regions(region_id) VALUES (99)")
    assert isinstance(caught.value.__cause__, psycopg.errors.ReadOnlySqlTransaction)
    after = await executor.execute("SELECT count(*) AS n FROM biz.regions WHERE region_id = 99")
    assert after.rows == before.rows


async def test_mcp_role_statement_timeout_is_set(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "mcp_ro") as conn:
        assert await conn.fetchval("SHOW statement_timeout") == "10s"


@pytest.mark.parametrize("database", ["insightpilot_app", "insightpilot_business"])
async def test_dblink_not_installed(database_stack: DatabaseStack, database: Database) -> None:
    async with connection(database_stack, "postgres", database) as conn:
        assert not await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('dblink', 'postgres_fdw')"
        )


@pytest.mark.parametrize("database", ["insightpilot_app", "insightpilot_business"])
async def test_vector_extension_not_enabled(
    database_stack: DatabaseStack, database: Database
) -> None:
    async with connection(database_stack, "postgres", database) as conn:
        assert not await conn.fetch("SELECT extname FROM pg_extension WHERE extname = 'vector'")


async def test_all_role_parameters(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "mcp_ro") as conn:
        assert await conn.fetchval("SHOW default_transaction_read_only") == "on"
        assert await conn.fetchval("SHOW idle_in_transaction_session_timeout") == "15s"
        assert await conn.fetchval("SHOW lock_timeout") == "2s"


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO biz.boundary_probe(value) VALUES (2)",
        "UPDATE biz.boundary_probe SET value = 2",
        "DELETE FROM biz.boundary_probe",
        "CREATE TABLE biz.forbidden(value integer)",
    ],
)
async def test_acl_rejects_writes_when_read_only_disabled(
    database_stack: DatabaseStack, statement: str
) -> None:
    async with connection(database_stack, "postgres") as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("SET LOCAL ROLE biz_owner")
            await conn.execute("CREATE TABLE biz.boundary_probe(value integer)")
            await conn.execute("INSERT INTO biz.boundary_probe VALUES (1)")
            # SET ROLE checks ACLs as mcp_ro without applying login defaults.
            await conn.execute("SET LOCAL ROLE mcp_ro")
            assert await conn.fetchval("SHOW transaction_read_only") == "off"
            assert await conn.fetchval("SELECT value FROM biz.boundary_probe") == 1
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(statement)
        finally:
            await transaction.rollback()


@pytest.mark.parametrize(
    ("database", "owner", "role", "schema"),
    [
        ("insightpilot_app", "app_owner", "app_rw", "public"),
        ("insightpilot_business", "biz_owner", "etl_rw", "biz"),
    ],
)
async def test_owner_defaults_allow_runtime_dml(
    database_stack: DatabaseStack, database: Database, owner: str, role: str, schema: str
) -> None:
    async with connection(database_stack, "postgres", database) as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            # Identifiers are the fixed parametrized literals above, never external input.
            await conn.execute(
                sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(owner)).as_string()
            )
            await conn.execute(
                sql.SQL("CREATE TABLE {}.default_probe(id serial, value integer)")
                .format(sql.Identifier(schema))
                .as_string()
            )
            await conn.execute(
                sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)).as_string()
            )
            await conn.execute(
                sql.SQL("INSERT INTO {}.default_probe(value) VALUES (1)")
                .format(sql.Identifier(schema))
                .as_string()
            )
            await conn.execute(
                sql.SQL("UPDATE {}.default_probe SET value = 2")
                .format(sql.Identifier(schema))
                .as_string()
            )
            expected = 2
            assert (
                await conn.fetchval(
                    sql.SQL("SELECT value FROM {}.default_probe")
                    .format(sql.Identifier(schema))
                    .as_string()
                )
                == expected
            )
            await conn.execute(
                sql.SQL("DELETE FROM {}.default_probe").format(sql.Identifier(schema)).as_string()
            )
            assert not await conn.fetchval(
                "SELECT has_schema_privilege(current_user, $1, 'CREATE')", schema
            )
        finally:
            await transaction.rollback()


async def test_runtime_roles_cannot_assume_owners(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "postgres") as conn:
        assert not await conn.fetch("""
            SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON r.oid = m.member
            WHERE r.rolname IN ('app_rw', 'etl_rw', 'mcp_ro', 'mcp_audit')
        """)
        assert not await conn.fetch("""
            SELECT 1 FROM pg_roles WHERE rolname IN ('app_rw', 'etl_rw', 'mcp_ro', 'mcp_audit', 'app_owner', 'biz_owner')
            AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)
        """)
        assert not await conn.fetch(
            "SELECT 1 FROM pg_roles WHERE rolname IN ('app_owner', 'biz_owner') AND rolcanlogin"
        )


async def test_special_character_password_authenticates(database_stack: DatabaseStack) -> None:
    settings = database_stack.settings
    password = quote_plus(settings.bootstrap.mcp_password.get_secret_value())
    url = f"postgresql://mcp_ro:{password}@127.0.0.1:{settings.db_host_port}/insightpilot_business"
    conn = await asyncpg.connect(url, timeout=5, command_timeout=5)
    try:
        assert await conn.fetchval("SELECT current_user") == "mcp_ro"
    finally:
        await conn.close(timeout=5)


async def test_bootstrap_twice_preserves_existing_data(database_stack: DatabaseStack) -> None:
    async with connection(database_stack, "postgres") as conn:
        # Persist this one sentinel in the isolated test volume to prove in-place bootstrap.
        await conn.execute("SET ROLE biz_owner")
        await conn.execute("CREATE TABLE biz.bootstrap_sentinel AS SELECT 42 AS value")
        await conn.execute("RESET ROLE")
        expected = 42
        before = await conn.fetch(
            "SELECT oid, datname FROM pg_database WHERE datname LIKE 'insightpilot_%' ORDER BY datname"
        )
        call = database_stack.call.model_copy(
            update={"command": Command.BOOTSTRAP, "arguments": []}
        )
        for _ in range(2):
            await asyncio.to_thread(execute, database_stack.docker, database_stack.settings, call)
            assert await conn.fetchval("SELECT value FROM biz.bootstrap_sentinel") == expected
            assert (
                await conn.fetch(
                    "SELECT oid, datname FROM pg_database WHERE datname LIKE 'insightpilot_%' ORDER BY datname"
                )
                == before
            )
    async with connection(database_stack, "mcp_ro") as conn:
        assert await conn.fetchval("SELECT value FROM biz.bootstrap_sentinel") == expected
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await conn.execute("INSERT INTO biz.bootstrap_sentinel VALUES (99)")
        await conn.execute("SET default_transaction_read_only = off")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("UPDATE biz.bootstrap_sentinel SET value = 99")
    await test_app_role_cannot_connect_to_business_db(database_stack)
    await test_all_role_parameters(database_stack)


async def test_bootstrap_applies_updated_password(database_stack: DatabaseStack) -> None:
    changed = database_stack.settings.model_copy(deep=True)
    changed.bootstrap.mcp_password = SecretStr(secrets.token_urlsafe(32) + "@:'/$")
    call = database_stack.call.model_copy(update={"command": Command.BOOTSTRAP, "arguments": []})
    try:
        await asyncio.to_thread(execute, database_stack.docker, changed, call)
        rotated = database_stack.model_copy(update={"settings": changed})
        async with connection(rotated, "mcp_ro") as conn:
            assert await conn.fetchval("SELECT current_user") == "mcp_ro"
        with pytest.raises(asyncpg.InvalidPasswordError):
            async with connection(database_stack, "mcp_ro"):
                pytest.fail("The old password was still accepted")
    finally:
        await asyncio.to_thread(execute, database_stack.docker, database_stack.settings, call)


def test_api_container_env_has_no_business_credential(database_stack: DatabaseStack) -> None:
    """Inspect only environment keys; never print rendered Compose secret values."""
    rendered = execute(
        database_stack.docker,
        database_stack.settings,
        database_stack.call.model_copy(
            update={"command": Command.CONFIG, "arguments": ["--format", "json"]}
        ),
    )
    services = json.loads(rendered)["services"]
    keys = set(services["api"]["environment"])
    assert environment_issues("api", keys).passed
    assert not services["mcp"].get("ports")
    assert services["mcp"]["networks"].keys() == {"backend"}
