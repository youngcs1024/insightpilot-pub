"""Ingest the declared corpus without API, business-database or SSH credentials."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import ClassVar, Self

from pydantic import Field, ValidationError, model_validator

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import DatabaseSettings, ModelRuntimeClientSettings
from app.core.errors import InsightPilotError
from app.core.settings_base import ProcessSettings, require_configuration
from app.db.session import Database
from app.retrieval.config import RetrievalSettings
from app.retrieval.ingestion_store import IngestionStore
from app.services.ingestion import IngestionService
from app.services.ingestion_config import IngestionSettings

MIN_CONNECTIONS = 2


class IngestionProcessSettings(ProcessSettings):
    """Only the application DML, storage and model-client settings are accepted."""

    process_name: ClassVar[str] = "ingest"
    database: DatabaseSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    model_runtime: ModelRuntimeClientSettings
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    @model_validator(mode="after")
    def independent_connections(self) -> Self:
        """Guard and publication require two simultaneous database connections."""
        require_configuration(
            self.database.pool_size + self.database.max_overflow >= MIN_CONNECTIONS,
            "Ingestion needs two connections",
        )
        return self


async def run(settings: IngestionProcessSettings, root: Path) -> int:
    """Close owned resources even if discovery, models or publication fail."""
    database = Database(settings.database)
    database.start()
    model = ModelRuntimeClient(settings.model_runtime)
    try:
        async with IngestionStore(settings.retrieval.milvus) as store:
            result = await IngestionService(
                database, store, model, settings.ingestion, settings.model_runtime
            ).ingest(root)
            print(
                f"{result.documents_changed} documents changed, {result.chunks_inserted} chunks inserted, {result.chunks_deleted} chunks deleted"
            )
            print(result.model_dump_json())
            return 0 if result.successful else 1
    finally:
        try:
            async with asyncio.timeout(10):
                await model.aclose()
        finally:
            async with asyncio.timeout(10):
                await database.aclose()


def main(argv: list[str] | None = None) -> int:
    """Emit only fixed safe failure messages and a nonzero operator exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(IngestionProcessSettings.load(), args.corpus))
    except ValidationError:
        print("Invalid ingestion configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
    except TimeoutError:
        print("Ingestion deadline exceeded; retry to finish pending cleanup.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
