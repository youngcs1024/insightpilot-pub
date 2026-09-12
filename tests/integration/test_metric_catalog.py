"""Published catalog and independently executed MCP examples on real PostgreSQL."""

# ruff: noqa: PLR2004 -- fixed catalog counts and independent numerical acceptance bounds.

import subprocess
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings, Settings
from app.core.errors import MetricCatalogError, MetricNotFound, UnsupportedGrain
from app.db.session import Database
from app.schemas.metric_resolution import MetricPatch, RegionScope
from app.schemas.metrics import Grain, MetricCatalog, MetricRenderContext, example_period
from app.services.metric_binding import build_binding
from app.services.metrics import (
    MetricService,
    render_catalog_block,
    render_expression,
    validate_grain,
)
from data.seed.metrics_loader import load_catalog
from tests.database_support import DatabaseStack
from tests.integration.catalog_support import (
    TransactionDatabase,
    catalog_migrated,
    cli_environment,
    client,
)
from tests.integration.mcp_support import mcp_endpoint, query
from tests.metric_resolution_support import request as resolution_request
from tests.metric_resolution_support import schema as resolution_schema
from tests.metric_support import expected_sql
from tests.seed_support import seed_database_stack, seed_directory, seeded

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
# Imported fixtures compose one isolated stack, with a separately running MCP process.
database_stack = seed_database_stack


@pytest.fixture(scope="module")
def seed_stack(database_stack: DatabaseStack) -> DatabaseStack:
    return database_stack


server_endpoint = mcp_endpoint
__all__ = ["catalog_migrated", "client", "seed_directory", "seeded"]


@pytest.fixture
async def metrics(
    catalog_migrated: None, database_stack: DatabaseStack
) -> AsyncIterator[MetricService]:
    config = DatabaseSettings(
        port=database_stack.settings.db_host_port,
        app_password=database_stack.settings.bootstrap.app_password,
    )
    database = Database(config)
    database.start()
    try:
        yield MetricService(database, config)
    finally:
        await database.aclose()


@pytest.fixture
async def owner(
    catalog_migrated: None, database_stack: DatabaseStack
) -> AsyncIterator[TransactionDatabase]:
    config = DatabaseSettings(
        port=database_stack.settings.db_host_port,
        app_user="postgres",
        app_password=database_stack.settings.postgres_superuser_password,
    )
    engine = create_async_engine(config.app_url, connect_args={"timeout": 5, "command_timeout": 10})
    try:
        async with engine.connect() as connection, connection.begin():
            sessions = async_sessionmaker(
                connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
            yield TransactionDatabase(sessions)
            await connection.rollback()
    finally:
        await engine.dispose()


async def test_six_metrics_seeded(metrics: MetricService) -> None:
    active = await metrics.list_active()
    assert len(active) == 6
    assert sorted(active, key=lambda d: d.key) == active
    authored = load_catalog(ROOT / "data/seed/metrics.yaml")
    snapshot = MetricCatalog.model_validate_json(
        (ROOT / "alembic/app/data/0007_metric_catalog.json").read_text()
    )
    assert snapshot == authored
    assert sorted(authored.definitions, key=lambda d: d.key) == active
    await metrics.validate_startup()


async def test_only_one_active_version_per_key(owner: TransactionDatabase) -> None:

    async with owner.session() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("""INSERT INTO metric_definitions
              SELECT key, 2, display_name, description, expression_template, base_tables,
              default_date_field, required_filters, supported_grains, examples, true, created_at, updated_at
              FROM metric_definitions WHERE key='gmv' AND version=1""")
            )


async def test_version_switch_preserves_history(
    owner: TransactionDatabase, settings: Settings
) -> None:
    service = MetricService(owner, settings.database)
    original = await service.get_version("gmv", 1)
    async with owner.session() as session, session.begin():
        await session.execute(text("UPDATE metric_definitions SET is_active=false WHERE key='gmv'"))
        await session.execute(
            text("""INSERT INTO metric_definitions
          SELECT key, 2, display_name, 'new published definition', expression_template, base_tables,
          default_date_field, required_filters, supported_grains, examples, true, created_at, updated_at
          FROM metric_definitions WHERE key='gmv' AND version=1""")
        )
    assert (await service.get_active("gmv")).version == 2
    historical = await service.get_version("gmv", 1)
    assert not historical.is_active
    assert historical.description == original.description
    assert historical.expression_template == original.expression_template


@pytest.mark.parametrize(("key", "version"), [("unknown", None), ("gmv", 99)])
async def test_unknown_key_raises(metrics: MetricService, key: str, version: int | None) -> None:
    call = metrics.get_active(key) if version is None else metrics.get_version(key, version)
    with pytest.raises(MetricNotFound):
        await call


async def test_unsupported_grain_raises(metrics: MetricService) -> None:
    for key in ("refund_rate", "refund_count"):
        with pytest.raises(UnsupportedGrain) as caught:
            validate_grain(await metrics.get_active(key), Grain.CATEGORY)
        assert "category" not in caught.value.supported


async def test_every_metric_example_sql_executes(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    for item in await metrics.list_active():
        for example in item.examples:
            actual = await query(client, example.sql)
            assert actual.row_count > 0
            assert not actual.result_truncated
        for grain in item.supported_grains:
            start, end = example_period()
            actual = await query(
                client,
                render_expression(
                    item, MetricRenderContext(period_start=start, period_end=end, grain=grain)
                ),
            )
            expected = await query(
                client, expected_sql(item.key, grain, start.isoformat(), end.isoformat())
            )
            assert actual.rows == expected.rows, (item.key, grain, actual.rows, expected.rows)


async def test_gmv_example_excludes_cancelled(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    item = await metrics.get_active("gmv")
    correct = await query(client, item.examples[0].sql)
    naive = await query(
        client,
        """SELECT SUM(o.gross_amount-o.discount_amount) FROM biz.orders o
      JOIN biz.customers c USING(customer_id) WHERE NOT c.is_test_account
      AND o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00'
      AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'""",
    )
    c, n = Decimal(str(correct.rows[0][1])), Decimal(str(naive.rows[0][0]))
    assert (n - c) / c >= Decimal(".05")


@pytest.mark.parametrize(
    "key", ["refund_rate", "aov", "refund_count", "order_count", "active_customer", "gmv"]
)
async def test_empty_period(
    metrics: MetricService, client: McpClient, seeded: object, key: str
) -> None:
    item = await metrics.get_active(key)
    context = MetricRenderContext.model_validate(
        {"period_start": "2024-01-01T00:00:00+08:00", "period_end": "2024-02-01T00:00:00+08:00"}
    )
    result = await query(client, render_expression(item, context))
    assert result.row_count == 1
    assert result.rows[0][1] == (None if key in {"refund_rate", "aov", "gmv"} else 0)


async def test_historical_category_null_discounts(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    context = MetricRenderContext.model_validate(
        {
            "period_start": "2025-03-01T00:00:00+08:00",
            "period_end": "2026-01-01T00:00:00+08:00",
            "grain": "category",
        }
    )
    result = await query(client, render_expression(await metrics.get_active("gmv"), context))
    expected = await query(
        client,
        expected_sql(
            "gmv", Grain.CATEGORY, context.period_start.isoformat(), context.period_end.isoformat()
        ),
    )
    assert result.rows == expected.rows
    missing = await query(
        client, "SELECT count(*) FROM biz.order_items WHERE item_discount IS NULL"
    )
    assert missing.rows[0][0] > 0


async def test_cross_month_refunds_and_duplicate_phone_are_present(
    client: McpClient, seeded: object
) -> None:
    result = await query(
        client,
        """SELECT
      count(*) FILTER (WHERE date_trunc('month', r.requested_at AT TIME ZONE 'Asia/Shanghai') <>
                             date_trunc('month', o.paid_at AT TIME ZONE 'Asia/Shanghai')) AS cross_month,
      count(*)-count(DISTINCT r.order_id) AS additional_refunds
      FROM biz.refunds r JOIN biz.orders o USING(order_id) WHERE r.status <> 'rejected'""",
    )
    assert all(Decimal(str(n)) > 0 for n in result.rows[0])
    result = await query(client, "SELECT count(*)-count(DISTINCT phone) FROM biz.customers")
    assert result.rows[0][0] > 0


async def test_startup_rejects_corrupt_persistence(
    owner: TransactionDatabase, settings: Settings
) -> None:
    service = MetricService(owner, settings.database)
    async with owner.session() as session, session.begin():
        await session.execute(
            text(
                "UPDATE metric_definitions SET supported_grains='[\"invented\"]'::jsonb WHERE key='gmv'"
            )
        )
    with pytest.raises(MetricCatalogError):
        await service.validate_startup()


async def test_runtime_role_cannot_mutate_catalog(metrics: MetricService) -> None:
    async with metrics._database.session() as session:
        permissions = (
            await session.execute(
                text("""SELECT
          has_table_privilege(current_user, 'metric_definitions', 'SELECT'),
          has_table_privilege(current_user, 'metric_definitions', 'INSERT'),
          has_table_privilege(current_user, 'metric_definitions', 'UPDATE'),
          has_table_privilege(current_user, 'metric_definitions', 'DELETE'),
          has_table_privilege(current_user, 'metric_definitions', 'TRUNCATE')""")
            )
        ).one()
        assert tuple(permissions) == (True, False, False, False, False)


def test_documented_cli(
    catalog_migrated: None,
    database_stack: DatabaseStack,
    server_endpoint: MCPSettings,
    settings: Settings,
    tmp_path: Path,
) -> None:
    migration_env = {
        "PATH": str(ROOT / ".venv/bin") + ":/usr/bin:/bin",
        "IP_MIGRATION__HOST": "127.0.0.1",
        "IP_MIGRATION__PORT": str(database_stack.settings.db_host_port),
        "IP_MIGRATION__PASSWORD": database_stack.settings.postgres_superuser_password.get_secret_value(),
    }
    migration = subprocess.run(  # noqa: S603 -- isolated operator process, no shared developer DB.
        [str(ROOT / ".venv/bin/alembic"), "-n", "app", "upgrade", "head"],
        env=migration_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (tmp_path / "metric-migration.txt").write_text(migration.stdout + migration.stderr)
    assert migration.returncode == 0, migration.stdout + migration.stderr
    env = cli_environment(settings, database_stack, server_endpoint)
    result = subprocess.run(  # noqa: S603 -- fixed CLI and isolated fixture credentials.
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.render_metric_catalog"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (tmp_path / "metric-catalog.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == render_catalog_block(
        load_catalog(ROOT / "data/seed/metrics.yaml").definitions
    )


@pytest.mark.parametrize("version", [0, -1])
async def test_database_rejects_nonpositive_versions(
    owner: TransactionDatabase, version: int
) -> None:
    async with owner.session() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("UPDATE metric_definitions SET version=:version WHERE key='gmv'"),
                {"version": version},
            )


async def test_startup_detects_duplicate_active_rows_if_index_missing(
    owner: TransactionDatabase, settings: Settings
) -> None:
    # Transactional DDL is rolled back; only this isolated fixture's catalog changes.
    async with owner.session() as session, session.begin():
        await session.execute(text("DROP INDEX uq_metric_definitions_active_key"))
        await session.execute(
            text("""INSERT INTO metric_definitions
          SELECT key, 2, display_name, description, expression_template, base_tables,
          default_date_field, required_filters, supported_grains, examples, true, created_at, updated_at
          FROM metric_definitions WHERE key='gmv' AND version=1""")
        )
    service = MetricService(owner, settings.database)
    with pytest.raises(MetricCatalogError):
        await service.validate_startup()
    with pytest.raises(MetricCatalogError):
        await service.get_active("gmv")


async def test_refund_only_period_does_not_report_zero(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    context = MetricRenderContext.model_validate(
        {
            "period_start": "2026-09-01T00:00:00+08:00",
            "period_end": "2026-10-01T00:00:00+08:00",
        }
    )
    count = await query(
        client, render_expression(await metrics.get_active("refund_count"), context)
    )
    assert count.rows[0][1] > 0
    rate = await query(client, render_expression(await metrics.get_active("refund_rate"), context))
    assert rate.rows == [["total", None]]


async def test_resolved_gmv_filter_and_expression_change_real_results(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    """One existing isolated stack supplies Step 2.6 numerical acceptance."""
    value = resolution_request(
        patch=MetricPatch(
            remove_filters=["o.status <> 'cancelled'"],
            expression="SUM(o.gross_amount)",
        )
    )
    value.definition = await metrics.get_active("gmv")
    binding = build_binding(value, resolution_schema()).binding
    actual = await query(client, binding.resolved_expression)
    expected = await query(
        client,
        """SELECT SUM(o.gross_amount)
        FROM biz.orders o JOIN biz.customers c USING (customer_id)
        WHERE o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00'
          AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'
          AND o.paid_at IS NOT NULL AND c.is_test_account = false""",
    )
    baseline = await query(client, value.definition.examples[0].sql)
    assert actual.rows[0][1] == expected.rows[0][0]
    assert Decimal(str(actual.rows[0][1])) > Decimal(str(baseline.rows[0][1]))


async def test_resolved_refund_payment_period_and_region_execute(
    metrics: MetricService, client: McpClient, seeded: object
) -> None:
    value = resolution_request("refund_rate", patch=MetricPatch(date_field="o.paid_at"))
    value.definition = await metrics.get_active("refund_rate")
    value.region_scope = RegionScope(region_ids=[3])
    binding = build_binding(value, resolution_schema()).binding
    actual = await query(client, binding.resolved_expression)
    expected = await query(
        client,
        """SELECT
        (SELECT COUNT(DISTINCT o.order_id)::numeric
         FROM biz.refunds r JOIN biz.orders o USING(order_id)
         JOIN biz.customers c USING(customer_id)
         WHERE o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00'
           AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'
           AND o.status <> 'cancelled' AND NOT c.is_test_account
           AND r.status <> 'rejected' AND o.region_id = 3)
        / NULLIF((SELECT COUNT(DISTINCT o.order_id)
         FROM biz.orders o JOIN biz.customers c USING(customer_id)
         WHERE o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00'
           AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'
           AND o.status <> 'cancelled' AND NOT c.is_test_account
           AND o.region_id = 3), 0) AS expected""",
    )
    assert actual.rows[0][1] == expected.rows[0][0]
    # Persistable binding points to the published version and exact requested scope.
    assert binding.definition_version == value.definition.version
    assert binding.region_scope.region_ids == [3]
