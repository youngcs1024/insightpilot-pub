"""Export application metadata into a deterministic, PII-safe MCP build resource."""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from app.core.config_models import Settings
from app.core.errors import DeadlineExceededError, InsightPilotError, SchemaMetadataError
from app.core.logging import setup_logging
from app.core.schema_artifact import artifact_json, build_artifact
from app.db.session import Database
from app.repositories.schema_metadata import SchemaMetadataRepository
from app.schemas.schema_tools import SchemaArtifact


async def export(database: Database, *, timeout_s: float) -> SchemaArtifact:
    """Read one bounded, consistent snapshot under the application credential only."""
    try:
        async with asyncio.timeout(timeout_s), database.session() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            return build_artifact(await SchemaMetadataRepository(session).read())
    except TimeoutError as exc:
        raise DeadlineExceededError() from exc


async def run(output: Path, *, check: bool) -> int:
    """Write a requested artifact, or compare without changing the committed resource."""
    settings = Settings.load()
    settings.observability.log_level = "WARNING"
    setup_logging(settings)
    database = Database(settings.database)
    database.start()
    try:
        artifact = await export(database, timeout_s=settings.schema_catalog.operation_timeout_s)
        source = artifact_json(artifact)
        if check:
            actual = await asyncio.to_thread(output.read_text, encoding="utf-8")
            if actual != source:
                raise SchemaMetadataError("The schema artifact differs from application metadata.")
        else:
            await asyncio.to_thread(output.write_text, source, encoding="utf-8")
        print(artifact.metadata_revision)
        return 0
    finally:
        await database.aclose()


def main() -> int:
    """Expose build/export and read-only CI drift-check modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mcp_server/data/schema_metadata.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.output, check=args.check))
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
        return 1
    except OSError:
        print("SCHEMA_METADATA_INVALID: Schema artifact could not be accessed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
