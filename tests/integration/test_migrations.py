"""Real PostgreSQL acceptance for independent histories and application constraints."""

from __future__ import annotations

import ast
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from alembic.autogenerate import produce_migrations
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from alembic import command
from app.db.base import Base
from app.db.models import Conversation, Turn, TurnRole, TurnStatus, User
from scripts import migration_environment
from scripts.migrate_all import migration_config
from scripts.migration_environment import (
    BUSINESS_METADATA,
    EXCLUDE_TABLES,
    MigrationError,
    configure_context,
)
from scripts.migration_settings import MigrationSettings, MigrationTarget

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tests.database_support import DatabaseStack

pytestmark = pytest.mark.integration


@pytest.fixture
async def engine(migrated: MigrationSettings) -> AsyncIterator[AsyncEngine]:
    """A bounded administrative inspection connection, never an API credential."""
    resource = create_async_engine(
        migrated.migration.url(MigrationTarget.APP),
        connect_args={"timeout": 5, "command_timeout": 30},
        hide_parameters=True,
    )
    try:
        yield resource
    finally:
        await resource.dispose()


def assert_empty_diff(connection: Connection, target: MigrationTarget) -> None:
    """Use the exact environment configuration rather than duplicating its exclusion rules."""
    config = migration_config(target)
    with EnvironmentContext(config, ScriptDirectory.from_config(config)) as environment:
        configure_context(environment, target, connection=connection)
        metadata = Base.metadata if target == MigrationTarget.APP else BUSINESS_METADATA
        migration = produce_migrations(environment.get_context(), metadata)
        assert migration.upgrade_ops is not None
        assert migration.upgrade_ops.is_empty(), migration.upgrade_ops.as_diffs()


async def test_upgrade_head_from_empty(migrated: MigrationSettings, engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert {"users", "conversations", "turns", "refresh_tokens", "alembic_version_app"} <= set(
            tables
        )
        columns = await connection.run_sync(lambda conn: inspect(conn).get_columns("turns"))
        clarification = next(column for column in columns if column["name"] == "clarification")
        assert clarification["nullable"]
        assert str(clarification["type"]) == "JSONB"
        assert (
            await connection.scalar(text("SELECT version_num FROM alembic_version_app"))
            == "0009_ingestion_registry"
        )
    business = create_async_engine(migrated.migration.url(MigrationTarget.BUSINESS))
    try:
        async with business.connect() as connection:
            tables = await connection.run_sync(
                lambda conn: inspect(conn).get_table_names(schema="biz")
            )
            assert set(tables) == {
                "alembic_version_biz",
                "regions",
                "promotions",
                "products",
                "customers",
                "orders",
                "order_items",
                "refunds",
                "inventory",
            }
            assert (
                await connection.scalar(text("SELECT version_num FROM biz.alembic_version_biz"))
                == "0001_business_schema"
            )
    finally:
        await business.dispose()


def test_registry_migration_roundtrip(migrated: MigrationSettings) -> None:
    """Downgrade only the registry, then prove a fresh upgrade matches ORM metadata."""
    config = migration_config(MigrationTarget.APP)
    command.downgrade(config, "0008_turn_clarification")

    async def inspect_registry(expected: bool) -> None:
        resource = create_async_engine(migrated.migration.url(MigrationTarget.APP))
        try:
            async with resource.connect() as connection:
                tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
                registry = {"documents", "chunks", "corpus_manifests"}
                assert registry <= set(tables) if expected else registry.isdisjoint(tables)
                assert "data_evidence" in tables
                if expected:
                    await connection.run_sync(assert_empty_diff, MigrationTarget.APP)
        finally:
            await resource.dispose()

    try:
        asyncio.run(inspect_registry(False))
    finally:
        command.upgrade(config, "head")
    asyncio.run(inspect_registry(True))


async def inspect_base(migrated: MigrationSettings, target: MigrationTarget) -> None:
    """Inspect one downgraded history without touching bootstrap or extension state."""
    resource = create_async_engine(migrated.migration.url(target))
    try:
        async with resource.connect() as connection:
            schema = "public" if target == MigrationTarget.APP else "biz"
            tables = await connection.run_sync(
                lambda conn: inspect(conn).get_table_names(schema=schema)
            )
            assert not {"users", "conversations", "turns", "refresh_tokens"} & set(tables)
            version = (
                "SELECT count(*) FROM public.alembic_version_app"
                if target == MigrationTarget.APP
                else "SELECT count(*) FROM biz.alembic_version_biz"
            )
            # Identifier is selected only from the two fixed version table names above.
            assert await connection.scalar(text(version)) == 0
            if target == MigrationTarget.APP:
                assert (
                    await connection.scalar(
                        text(
                            "SELECT count(*) FROM pg_type WHERE typname IN ('turn_role', 'turn_status')"
                        )
                    )
                    == 0
                )
                assert (
                    await connection.scalar(
                        text("SELECT count(*) FROM pg_extension WHERE extname = 'citext'")
                    )
                    == 1
                )
    finally:
        await resource.dispose()


def test_downgrade_to_base(migrated: MigrationSettings) -> None:
    try:
        for target in MigrationTarget:
            command.downgrade(migration_config(target), "base")
        for target in MigrationTarget:
            asyncio.run(inspect_base(migrated, target))
    finally:
        for target in MigrationTarget:
            command.upgrade(migration_config(target), "head")


@pytest.mark.parametrize("target", list(MigrationTarget))
async def test_autogenerate_produces_empty_diff(
    migrated: MigrationSettings, target: MigrationTarget
) -> None:
    resource = create_async_engine(migrated.migration.url(target))
    try:
        async with resource.connect() as connection:
            await connection.run_sync(assert_empty_diff, target)
    finally:
        await resource.dispose()


async def test_checkpoint_tables_excluded(engine: AsyncEngine) -> None:
    # These deliberately incomplete stand-ins must roll back on connection close.
    # Committing them breaks real checkpointer setup when this module runs alone.
    async with engine.connect() as connection:
        for name in sorted(EXCLUDE_TABLES):
            # Closed module-owned identifiers; no caller input participates in SQL.
            await connection.execute(
                text('CREATE TABLE IF NOT EXISTS "' + name + '" (id integer PRIMARY KEY)')
            )
        await connection.run_sync(assert_empty_diff, MigrationTarget.APP)
        tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert set(tables) >= EXCLUDE_TABLES


@pytest.mark.parametrize("target", list(MigrationTarget))
def test_real_autogenerate_file_is_empty(
    migrated: MigrationSettings, tmp_path: Path, target: MigrationTarget
) -> None:
    config = migration_config(target)
    location = config.get_main_option("script_location")
    assert location is not None
    temporary = tmp_path / target.value
    shutil.copytree(location, temporary)
    config.set_main_option("script_location", str(temporary))
    config.set_main_option("prepend_sys_path", str(Path.cwd()))
    temporary_ini = tmp_path / "alembic.ini"
    with temporary_ini.open("w") as output:
        config.file_config.write(output)
    result = subprocess.run(  # noqa: S603 -- fixed CLI, fixture-controlled config; no shell.
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(temporary_ini),
            "-n",
            target.value,
            "revision",
            "--autogenerate",
            "-m",
            "should be empty",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generated = list((temporary / "versions").glob("*_should_be_empty.py"))
    assert len(generated) == 1
    module = ast.parse(generated[0].read_text())
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    assert {"upgrade", "downgrade"} <= functions.keys()
    for name in ("upgrade", "downgrade"):
        statements = functions[name].body[1:]  # Generated function docstring is first.
        assert len(statements) == 1
        assert isinstance(statements[0], ast.Pass)


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Test all DML as app_rw inside a transaction rolled back after the test."""
    async with engine.connect() as connection, connection.begin():
        await connection.execute(text("SET LOCAL ROLE app_rw"))
        async with AsyncSession(bind=connection, expire_on_commit=False) as resource:
            yield resource
        await connection.rollback()


async def conversation(session: AsyncSession) -> Conversation:
    """Persist one independent user and conversation in the surrounding transaction."""
    user = User(
        email=f"{uuid4().hex}@example.invalid", hashed_password=uuid4().hex, display_name="Test"
    )
    session.add(user)
    await session.flush()
    result = Conversation(user_id=user.id, title="Migration acceptance")
    session.add(result)
    await session.flush()
    return result


async def test_email_case_insensitive_unique(session: AsyncSession) -> None:
    convo = await conversation(session)
    user = await session.get(User, convo.user_id)
    assert user is not None
    assert user.is_active
    duplicate = User(
        email=user.email.upper(), hashed_password=uuid4().hex, display_name="Duplicate"
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        await session.flush()


@pytest.mark.parametrize("constraint", ["seq", "idempotency_key", "running"])
async def test_turn_uniqueness(session: AsyncSession, constraint: str) -> None:
    convo = await conversation(session)
    status = TurnStatus.RUNNING if constraint == "running" else TurnStatus.SUCCEEDED
    session.add(
        Turn(
            conversation_id=convo.id,
            seq=1,
            role=TurnRole.ASSISTANT,
            content="",
            status=status,
            idempotency_key="first",
        )
    )
    await session.flush()
    session.add(
        Turn(
            conversation_id=convo.id,
            seq=1 if constraint == "seq" else 2,
            role=TurnRole.ASSISTANT,
            content="",
            status=status,
            idempotency_key="first" if constraint == "idempotency_key" else "second",
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_null_keys_enums_defaults_and_json(session: AsyncSession) -> None:
    convo = await conversation(session)
    assert convo.summary_through_seq == 0
    assert convo.summary is None
    assert convo.archived_at is None
    assert convo.created_at.tzinfo is not None
    for seq, status in enumerate(TurnStatus, start=1):
        session.add(
            Turn(
                conversation_id=convo.id,
                seq=seq,
                role=TurnRole.USER if seq == 1 else TurnRole.ASSISTANT,
                content="",
                status=status,
                token_usage={"input_tokens": 3, "details": {"cached": 1}},
            )
        )
    await session.flush()
    rows = (
        await session.execute(
            text(
                "SELECT role::text, status::text, idempotency_key, token_usage FROM turns WHERE conversation_id = :id ORDER BY seq"
            ),
            {"id": convo.id},
        )
    ).all()
    assert [row.status for row in rows] == [status.value for status in TurnStatus]
    assert rows[0].role == "user"
    assert rows[1].role == "assistant"
    assert all(row.idempotency_key is None for row in rows)
    assert rows[0].token_usage == {"input_tokens": 3, "details": {"cached": 1}}


async def test_foreign_keys_cascade(session: AsyncSession) -> None:
    convo = await conversation(session)
    turn = Turn(
        conversation_id=convo.id,
        seq=1,
        role=TurnRole.USER,
        content="hello",
        status=TurnStatus.SUCCEEDED,
    )
    session.add(turn)
    await session.flush()
    await session.execute(text("DELETE FROM users WHERE id = :id"), {"id": convo.user_id})
    assert (
        await session.scalar(
            text("SELECT count(*) FROM conversations WHERE id = :id"), {"id": convo.id}
        )
        == 0
    )
    assert (
        await session.scalar(text("SELECT count(*) FROM turns WHERE id = :id"), {"id": turn.id})
        == 0
    )


async def test_object_owners_and_runtime_permissions(
    engine: AsyncEngine, migrated: MigrationSettings
) -> None:
    async with engine.connect() as connection:
        owners = (
            (
                await connection.execute(
                    text(
                        "SELECT tableowner FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('users','conversations','turns','alembic_version_app')"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert owners == ["app_owner"] * 4
        assert await connection.scalar(
            text("SELECT has_table_privilege('app_rw', 'users', 'SELECT,INSERT,UPDATE,DELETE')")
        )
        assert not await connection.scalar(
            text("SELECT has_schema_privilege('app_rw', 'public', 'CREATE')")
        )
        assert not await connection.scalar(
            text("SELECT has_database_privilege('app_rw', 'insightpilot_business', 'CONNECT')")
        )
    business = create_async_engine(migrated.migration.url(MigrationTarget.BUSINESS))
    try:
        async with business.connect() as connection:
            assert (
                await connection.scalar(
                    text(
                        "SELECT tableowner FROM pg_tables WHERE schemaname = 'biz' AND tablename = 'alembic_version_biz'"
                    )
                )
                == "biz_owner"
            )
            assert not await connection.scalar(
                text("SELECT has_schema_privilege('etl_rw', 'biz', 'CREATE')")
            )
            assert not await connection.scalar(
                text("SELECT has_table_privilege('mcp_ro', 'biz.alembic_version_biz', 'INSERT')")
            )
    finally:
        await business.dispose()


@pytest.mark.parametrize("failure", ["statement_timeout", "sql_error"])
async def test_migration_failure_rolls_back_without_retry(
    migrated: MigrationSettings,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Real DDL before a failed statement must not survive the single attempted transaction."""

    attempts = 0

    def failing_revision(connection: Connection, target: MigrationTarget) -> None:
        nonlocal attempts
        attempts += 1
        with connection.begin():
            connection.execute(text("SET LOCAL ROLE app_owner"))
            connection.execute(text("CREATE TABLE step13_must_rollback (id integer)"))
            connection.execute(
                text("SELECT pg_sleep(1)" if failure == "statement_timeout" else "SELECT 1 / 0")
            )

    monkeypatch.setattr(migration_environment, "migrate_connection", failing_revision)
    bounded = migrated.model_copy(
        update={"migration": migrated.migration.model_copy(update={"command_timeout_s": 0.05})}
    )
    with pytest.raises(migration_environment.MigrationError) as caught:
        await migration_environment.run_online(bounded, MigrationTarget.APP)
    assert not caught.value.retryable
    assert caught.value.__suppress_context__
    assert attempts == 1
    resource = create_async_engine(migrated.migration.url(MigrationTarget.APP))
    try:
        async with resource.connect() as connection:
            assert (
                await connection.scalar(text("SELECT to_regclass('public.step13_must_rollback')"))
                is None
            )
    finally:
        await resource.dispose()


@pytest.mark.parametrize("credential", ["runtime_role", "wrong_password"])
def test_migration_rejects_invalid_credentials_safely(
    migrated: MigrationSettings,
    migration_stack: DatabaseStack,
    monkeypatch: pytest.MonkeyPatch,
    credential: str,
) -> None:

    if credential == "runtime_role":
        monkeypatch.setenv("IP_MIGRATION__USER", "app_rw")
        monkeypatch.setenv(
            "IP_MIGRATION__PASSWORD",
            migration_stack.settings.bootstrap.app_password.get_secret_value(),
        )
    else:
        monkeypatch.setenv("IP_MIGRATION__PASSWORD", "deliberately-invalid")
    with pytest.raises(MigrationError) as caught:
        command.upgrade(migration_config(MigrationTarget.APP), "head")
    assert str(caught.value) == "Database migration failed."
    assert caught.value.__suppress_context__


def test_cli_upgrade_and_make_migrate(migrated: MigrationSettings) -> None:
    """Run documented online commands against the fixture databases, never the developer stack."""
    for target in MigrationTarget:
        result = subprocess.run(  # noqa: S603 -- fixed executable and closed migration targets.
            [sys.executable, "-m", "alembic", "-n", target.value, "upgrade", "head"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "migration_completed" in result.stdout
    executable = shutil.which("make")
    assert executable is not None
    result = subprocess.run(  # noqa: S603 -- fixed project Makefile and safe command.
        [executable, "migrate", "ENV=test"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("migration_completed") == len(MigrationTarget)
