"""Real application PostgreSQL joined with an independently running MCP process."""

# ruff: noqa: PLR2004 -- fixed eight-table, 52-column acceptance contract.

import asyncio
import secrets
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings, SchemaCatalogSettings, Settings
from app.core.deadline import Deadline
from app.core.errors import McpUnavailableError, SchemaDriftError, ValidationError
from app.schemas.schema_catalog import (
    BUSINESS_TABLES,
    BusinessSchemaArguments,
    BusinessSchemaResponse,
    DriftKind,
)
from app.services.schema_catalog import SchemaCatalogService
from tests.database_support import DatabaseStack
from tests.integration.catalog_support import (
    TransactionDatabase,
    catalog_migrated,
    cli_environment,
    client,
)
from tests.integration.mcp_support import mcp_endpoint as existing_endpoint
from tests.seed_support import seed_database_stack

pytestmark = pytest.mark.integration
__all__ = ["catalog_migrated", "client"]
ROOT = Path(__file__).resolve().parents[2]
database_stack = seed_database_stack
server_endpoint = existing_endpoint


@pytest.fixture
async def database(
    catalog_migrated: None, database_stack: DatabaseStack
) -> AsyncIterator[TransactionDatabase]:
    settings = DatabaseSettings(
        port=database_stack.settings.db_host_port,
        app_password=database_stack.settings.bootstrap.app_password,
    )
    engine = create_async_engine(settings.app_url)
    try:
        async with engine.connect() as connection, connection.begin():
            # Outer rollback isolates deliberate missing/stale/PII metadata mutations.
            await connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            sessions = async_sessionmaker(
                connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
            yield TransactionDatabase(sessions)
            await connection.rollback()
    finally:
        await engine.dispose()


@pytest.fixture
async def catalog(database: TransactionDatabase, client: McpClient) -> SchemaCatalogService:
    return SchemaCatalogService(database, client, SchemaCatalogSettings())


async def change(database: TransactionDatabase, sql: str) -> None:
    async with database.session() as session, session.begin():
        await session.execute(text(sql))


async def test_every_business_column_has_metadata(catalog: SchemaCatalogService) -> None:
    report = await catalog.validate_against_live_schema()
    assert report.valid, report.model_dump_json(indent=2)
    block = await catalog.render()
    assert block.count("Table: ") == 8
    assert block.count("  - ") >= 52


async def test_render_includes_allowed_values(catalog: SchemaCatalogService) -> None:
    assert "cancelled=已取消" in await catalog.render(["biz.orders"])


async def test_render_excludes_pii_sample_values(
    catalog: SchemaCatalogService, database: TransactionDatabase
) -> None:
    await change(
        database,
        "UPDATE schema_metadata SET sample_values='[\"PII_SENTINEL\"]' WHERE table_name='biz.customers' AND column_name IN ('phone','email')",
    )
    assert "PII_SENTINEL" not in await catalog.render()


async def test_validate_detects_missing_metadata(
    catalog: SchemaCatalogService, database: TransactionDatabase
) -> None:
    await change(
        database,
        "DELETE FROM schema_metadata WHERE table_name='biz.orders' AND column_name='paid_at'",
    )
    report = await catalog.validate_against_live_schema()
    assert any(
        d.kind == DriftKind.MISSING_METADATA and d.column_name == "paid_at"
        for d in report.differences
    )
    with pytest.raises(SchemaDriftError):
        await catalog.render()


async def test_validate_detects_stale_metadata(
    catalog: SchemaCatalogService, database: TransactionDatabase
) -> None:
    await change(
        database,
        "UPDATE schema_metadata SET column_name='obsolete' WHERE table_name='biz.orders' AND column_name='paid_at'",
    )
    report = await catalog.validate_against_live_schema()
    assert any(
        d.kind == DriftKind.STALE_METADATA and d.column_name == "obsolete"
        for d in report.differences
    )


async def test_render_cached_and_invalidated_on_revision_change(
    catalog: SchemaCatalogService, database: TransactionDatabase
) -> None:
    old = await catalog.render()
    await change(
        database, "UPDATE schema_tables SET description='REVISED' WHERE table_name='biz.orders'"
    )
    assert await catalog.render() == old
    await change(database, "UPDATE alembic_version_app SET version_num='schema-revision-test'")
    assert "REVISED" in await catalog.render()


async def test_ttl_refresh_and_validation_bypasses_cache(
    database: TransactionDatabase, client: McpClient
) -> None:
    now = [0.0]
    catalog = SchemaCatalogService(
        database, client, SchemaCatalogSettings(ttl_s=1), clock=lambda: now[0]
    )
    await catalog.render()
    await change(
        database,
        "UPDATE schema_metadata SET sql_type='numeric(14,2)' WHERE table_name='biz.orders' AND column_name='gross_amount'",
    )
    await catalog.render()  # Valid until TTL (manual edits have no migration revision).
    now[0] = 2
    with pytest.raises(SchemaDriftError):
        await catalog.render()
    assert not (await catalog.validate_against_live_schema()).valid


@pytest.mark.parametrize(
    ("assignment", "kind"),
    [
        ("sql_type='timestamp without time zone'", DriftKind.TYPE_MISMATCH),
        ("nullable=false", DriftKind.NULLABILITY),
        ("is_primary_key=true", DriftKind.PRIMARY_KEY),
        ("fk_table='biz.products', fk_column='product_id'", DriftKind.FOREIGN_KEY),
        ("constraints='[]'", DriftKind.CONSTRAINTS),
    ],
)
async def test_validate_detects_structural_drift(
    catalog: SchemaCatalogService, database: TransactionDatabase, assignment: str, kind: DriftKind
) -> None:
    # assignment is a closed test parameter, never external input.
    await change(
        database,
        "UPDATE schema_metadata SET "  # noqa: S608 -- closed test assignments.
        + assignment
        + " WHERE table_name='biz.orders' AND column_name='paid_at'",
    )
    assert kind in {d.kind for d in (await catalog.validate_against_live_schema()).differences}


async def test_revision_hint_and_boundary(client: McpClient) -> None:
    live = await client.get_business_schema(
        BusinessSchemaArguments(), deadline=Deadline(time.monotonic() + 15)
    )
    assert {t.table_name for t in live.tables} == set(BUSINESS_TABLES)
    assert sum(len(t.columns) for t in live.tables) == 52
    assert "sample_values" not in live.model_dump_json()
    result = await client.get_business_schema(
        BusinessSchemaArguments(known_revision=live.revision),
        deadline=Deadline(time.monotonic() + 15),
    )
    assert result.unchanged
    assert result.tables == []


async def test_invalid_selection_and_stable_order(catalog: SchemaCatalogService) -> None:
    for selection in ([], ["biz.orders", "pg_catalog.pg_user"], ["ops.seed_manifest"]):
        with pytest.raises(ValidationError):
            await catalog.render(selection)
    assert await catalog.render(
        ["biz.orders", "biz.regions", "biz.orders"]
    ) == await catalog.render(["biz.regions", "biz.orders"])


async def test_refresh_failure_never_returns_stale(
    catalog: SchemaCatalogService, client: McpClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await catalog.render()

    async def failed(*args: object, **kwargs: object) -> None:
        raise McpUnavailableError()

    monkeypatch.setattr(client, "get_business_schema", failed)
    with pytest.raises(McpUnavailableError):
        await catalog.render()
    assert catalog._cache is None


async def test_concurrent_refresh_is_coalesced(
    catalog: SchemaCatalogService, client: McpClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = client.get_business_schema

    async def counted(
        args: BusinessSchemaArguments, *, deadline: Deadline
    ) -> BusinessSchemaResponse:
        calls.append(args.known_revision)
        return await original(args, deadline=deadline)

    monkeypatch.setattr(client, "get_business_schema", counted)
    blocks = await asyncio.gather(*(catalog.render() for _ in range(5)))
    assert len(set(blocks)) == 1
    assert calls.count(None) == 1


def test_cli_drift_gate_and_render(
    catalog_migrated: None,
    database_stack: DatabaseStack,
    server_endpoint: MCPSettings,
    settings: Settings,
    tmp_path: Path,
) -> None:
    env = cli_environment(settings, database_stack, server_endpoint)
    for module in ("scripts.check_schema_metadata", "scripts.render_schema_catalog"):
        result = subprocess.run(  # noqa: S603 -- fixed project CLI modules.
            [str(ROOT / ".venv/bin/python"), "-m", module],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (tmp_path / (module + ".txt")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "differences" if module.endswith("metadata") else "Table: biz.orders"
        ) in result.stdout


@asynccontextmanager
async def operator_change(
    stack: DatabaseStack,
    database_name: str,
    forward: list[str],
    reverse: list[str],
) -> AsyncIterator[None]:
    """Commit explicit changes only in the fixture database, then restore them."""
    settings = DatabaseSettings(
        port=stack.settings.db_host_port,
        app_db=database_name,
        app_user="postgres",
        app_password=stack.settings.postgres_superuser_password,
    )
    engine = create_async_engine(settings.app_url)
    try:
        await execute_statements(engine, forward)
        try:
            yield
        finally:
            await execute_statements(engine, reverse)
    finally:
        await engine.dispose()


async def test_business_revision_change_invalidates_cache(
    catalog: SchemaCatalogService, database_stack: DatabaseStack
) -> None:
    await catalog.render()
    async with operator_change(
        database_stack,
        "insightpilot_business",
        [
            "ALTER TABLE biz.products ALTER COLUMN sku TYPE varchar(40)",
            "UPDATE biz.alembic_version_biz SET version_num='business-changed'",
        ],
        [
            "ALTER TABLE biz.products ALTER COLUMN sku TYPE varchar(32)",
            "UPDATE biz.alembic_version_biz SET version_num='0001_business_schema'",
        ],
    ):
        with pytest.raises(SchemaDriftError):
            await catalog.render()
        assert catalog._cache is None


async def test_validation_reads_actual_ddl_without_revision_change(
    catalog: SchemaCatalogService, database_stack: DatabaseStack
) -> None:
    await catalog.render()
    async with operator_change(
        database_stack,
        "insightpilot_business",
        [
            "ALTER TABLE biz.customers ALTER COLUMN registered_at DROP NOT NULL",
        ],
        [
            "ALTER TABLE biz.customers ALTER COLUMN registered_at SET NOT NULL",
        ],
    ):
        report = await catalog.validate_against_live_schema()
        assert any(
            d.kind == DriftKind.NULLABILITY and d.column_name == "registered_at"
            for d in report.differences
        )


async def test_cli_fails_on_drift_and_dependency_failure(
    catalog_migrated: None,
    database_stack: DatabaseStack,
    server_endpoint: MCPSettings,
    settings: Settings,
) -> None:
    env = cli_environment(settings, database_stack, server_endpoint)
    async with operator_change(
        database_stack,
        "insightpilot_app",
        [
            "UPDATE schema_metadata SET sql_type='bigint' WHERE table_name='biz.orders' AND column_name='paid_at'",
        ],
        [
            "UPDATE schema_metadata SET sql_type='timestamp with time zone' WHERE table_name='biz.orders' AND column_name='paid_at'",
        ],
    ):
        result = await asyncio.to_thread(
            subprocess.run,
            [str(ROOT / ".venv/bin/python"), "-m", "scripts.check_schema_metadata"],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 1
        assert '"kind": "type_mismatch"' in result.stdout
    env["IP_MCP__AUTH_TOKEN"] = secrets.token_urlsafe(32)
    result = await asyncio.to_thread(
        subprocess.run,
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.check_schema_metadata"],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 1
    assert any(code in result.stdout for code in ("AUTHENTICATION_ERROR", "MCP_INVALID_RESULT"))
    assert '"differences": []' not in result.stdout


async def execute_statements(engine: AsyncEngine, statements: list[str]) -> None:
    async with engine.begin() as connection:
        for statement in statements:
            await connection.execute(text(statement))


async def test_metric_schema_snapshot_is_detached_and_drift_checked(
    catalog: SchemaCatalogService, database: TransactionDatabase
) -> None:
    snapshot = await catalog.snapshot()
    assert sum(len(t.columns) for t in snapshot.tables) == 52
    snapshot.tables.clear()
    assert len((await catalog.snapshot()).tables) == 8
    await change(
        database,
        "UPDATE schema_metadata SET sql_type='invented' WHERE table_name='biz.orders' AND column_name='paid_at'",
    )
    catalog.clear()
    with pytest.raises(SchemaDriftError):
        await catalog.snapshot()
