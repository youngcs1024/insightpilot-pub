"""Failure rollback, competing first imports and no-manifest protection on real SQL."""

# Frozen small-fixture row count.
# ruff: noqa: PLR2004

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from alembic import command
from data.seed.contracts import Dataset, Manifest, Parameters, SeedConflictError
from data.seed.files import export, read
from data.seed.generation import generate
from data.seed.schema import SEED_MANIFEST, TABLES
from scripts import seed
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget
from scripts.seed_settings import SeedDatabaseSettings, SeedSettings
from tests.database_support import DatabaseStack
from tests.seed_support import seed_database_stack

pytestmark = pytest.mark.integration
import_stack = seed_database_stack


@pytest.fixture(scope="module")
def empty_settings(import_stack: DatabaseStack) -> SeedSettings:
    stack = import_stack.settings
    settings = MigrationSettings(
        _env_file=None,
        migration={"port": stack.db_host_port, "password": stack.postgres_superuser_password},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MigrationSettings, "load", classmethod(lambda cls: settings))
        command.upgrade(migration_config(MigrationTarget.BUSINESS), "head")
    return SeedSettings(
        _env_file=None,
        seed=SeedDatabaseSettings(port=stack.db_host_port, password=stack.bootstrap.etl_password),
    )


async def test_atomic_failure_then_competing_first_imports(
    empty_settings: SeedSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "seed"
    export(generate(Parameters(orders=100, months=1)), directory)
    manifest, dataset = read(directory)
    original = seed.load_transaction

    async def fail_after_rows(
        connection: AsyncConnection, identity: Manifest, rows: Dataset, batch_size: int
    ) -> str:
        await original(connection, identity, rows, batch_size)
        raise SQLAlchemyError("Injected failure before commit")

    engine = create_async_engine(empty_settings.seed.url)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(seed, "load_transaction", fail_after_rows)
            with pytest.raises(seed.SeedDatabaseError):
                await seed.import_seed(directory, empty_settings)
        async with engine.connect() as connection:
            for table in (*TABLES, SEED_MANIFEST):
                assert await connection.scalar(select(func.count()).select_from(table)) == 0
            # Existing unregistered data must not be adopted or overwritten.
            await connection.execute(TABLES[0].insert(), dataset.tables[0][0].model_dump())
            with pytest.raises(SeedConflictError):
                await original(connection, manifest, dataset, 100)
            await connection.rollback()
        outcomes = await asyncio.gather(
            seed.import_seed(directory, empty_settings), seed.import_seed(directory, empty_settings)
        )
        assert sorted(result.outcome for result in outcomes) == ["imported", "unchanged"]
        async with engine.connect() as connection:
            assert await connection.scalar(select(func.count()).select_from(TABLES[4])) == 100
            assert await connection.scalar(select(func.count()).select_from(SEED_MANIFEST)) == 1
    finally:
        await engine.dispose()
