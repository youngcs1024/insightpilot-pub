"""Create-only, atomic business seed import as the isolated ETL role."""

import argparse
import asyncio
from pathlib import Path

import structlog
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.core.errors import DatabaseError, InsightPilotError
from app.core.settings_base import PROJECT_ROOT
from data.seed.contracts import Dataset, Manifest, SeedConflictError, SeedResult
from data.seed.files import read
from data.seed.schema import SEED_MANIFEST, TABLES
from scripts.seed_settings import SeedSettings

LOGGER = structlog.get_logger()


class SeedDatabaseError(DatabaseError):
    """Import failed; the transaction rolls back, without automatic write retries."""

    code = "SEED_DATABASE_FAILED"


async def load_transaction(
    connection: AsyncConnection, manifest: Manifest, dataset: Dataset, batch_size: int
) -> str:
    """The caller owns the transaction; this adapter never commits."""
    await connection.execute(text("SELECT pg_advisory_xact_lock(2101, 1)"))
    existing = (
        await connection.execute(
            select(SEED_MANIFEST.c.manifest).where(SEED_MANIFEST.c.dataset_id == "business")
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing != manifest.model_dump(mode="json"):
            raise SeedConflictError("Database manifest differs; use an isolated empty database.")
        return "unchanged"
    for table in TABLES:
        if (await connection.execute(select(table).limit(1))).first() is not None:
            raise SeedConflictError("Business data exists without a matching manifest.")
    for table, rows in zip(TABLES, dataset.tables, strict=True):
        for offset in range(0, len(rows), batch_size):
            await connection.execute(
                table.insert(), [row.model_dump() for row in rows[offset : offset + batch_size]]
            )
    await connection.execute(
        SEED_MANIFEST.insert().values(
            dataset_id="business", manifest=manifest.model_dump(mode="json")
        )
    )
    return "imported"


async def import_seed(directory: Path, settings: SeedSettings) -> SeedResult:
    """Validate files, serialize competing imports, then commit data and identity together."""
    manifest, dataset = await asyncio.to_thread(read, directory)
    config = settings.seed
    engine = create_async_engine(
        config.url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={
            "timeout": config.connect_timeout_s,
            "command_timeout": config.command_timeout_s,
            "server_settings": {
                "statement_timeout": str(int(config.command_timeout_s * 1000)),
                "lock_timeout": str(int(config.lock_timeout_s * 1000)),
            },
        },
    )
    outcome = "imported"
    try:
        async with engine.begin() as connection:
            outcome = await load_transaction(connection, manifest, dataset, config.batch_size)
    except (SQLAlchemyError, OSError, TimeoutError) as exc:
        LOGGER.exception("seed_import_failed", exception_type=type(exc).__name__, exc_info=False)
        raise SeedDatabaseError("Seed import failed; no automatic write replay.") from None
    finally:
        await engine.dispose()
    LOGGER.info("seed_import_completed", outcome=outcome, orders=manifest.parameters.orders)
    return SeedResult(outcome=outcome, manifest=manifest, directory=directory)


def main() -> None:
    """Import a generated directory using process-scoped credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/seed/out")
    args = parser.parse_args()
    try:
        result = asyncio.run(import_seed(args.input, SeedSettings.load()))
    except (InsightPilotError, ValidationError, OSError):
        parser.exit(
            1,
            "Seed import failed; check seed files, ETL configuration and existing database state.\n",
        )
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
