"""Check registry/index consistency; --fix repairs reproducible committed versions only."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import Field, ValidationError, model_validator

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import DatabaseSettings, ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.core.errors import IngestionConfigurationError, InsightPilotError
from app.core.settings_base import ProcessSettings, require_configuration
from app.db.session import Database
from app.retrieval.config import RetrievalSettings
from app.retrieval.consistency_store import ConsistencyStore
from app.schemas.ingestion import ActiveManifest, PreparedDocument, VectorRow
from app.schemas.mcp import Contract
from app.services.consistency import ConsistencyService
from app.services.ingestion import IngestionService
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import IngestionPlan

MIN_CONNECTIONS = 2


class ConsistencyProcessSettings(ProcessSettings):
    """Read-only checks require no corpus, model token, API or business credentials."""

    process_name: ClassVar[str] = "consistency"
    database: DatabaseSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    model_runtime: ModelRuntimeClientSettings | None = None

    @model_validator(mode="after")
    def independent_connections(self) -> Self:
        """The guard must outlive the independent registry commit."""
        require_configuration(
            self.database.pool_size + self.database.max_overflow >= MIN_CONNECTIONS,
            "Consistency maintenance needs two connections",
        )
        return self


class ConsistencyFailure(Contract):
    """An incomplete scan can never masquerade as an empty successful report."""

    schema_version: Literal[1] = 1
    successful: Literal[False] = False
    code: str


class RepairEncoder:
    """Create and close the model client only when reconstruction requires encoding."""

    def __init__(
        self, settings: ConsistencyProcessSettings, database: Database, store: ConsistencyStore
    ) -> None:
        self.settings = settings
        self.database = database
        self.store = store

    async def __call__(
        self, prepared: list[PreparedDocument], manifest: ActiveManifest, deadline: Deadline
    ) -> list[VectorRow]:
        model_settings = self.settings.model_runtime
        if model_settings is None:
            raise IngestionConfigurationError()
        model = ModelRuntimeClient(model_settings)
        try:
            service = IngestionService(
                self.database, self.store, model, self.settings.ingestion, model_settings
            )
            rows, _ = await service.encode(IngestionPlan(changed=prepared), manifest, deadline)
            return rows
        finally:
            async with asyncio.timeout(10):
                await model.aclose()


async def run(settings: ConsistencyProcessSettings, root: Path | None) -> int:
    """Own resources, lazily opening model HTTP only for actual reconstruction."""
    database = Database(settings.database)
    database.start()
    try:
        async with ConsistencyStore(settings.retrieval.milvus) as store:
            result = await ConsistencyService(
                database,
                store,
                settings.ingestion,
                RepairEncoder(settings, database, store) if settings.model_runtime else None,
            ).check(root=root)
            print(
                f"{len(result.remaining)} drift findings; "
                f"{result.chunks_inserted} chunks inserted; "
                f"{result.chunks_deleted} chunks deleted; "
                f"{len(result.blocked)} blocked; {len(result.cleanup_pending)} cleanup pending"
            )
            print(result.model_dump_json())
            return 0 if result.successful else 1
    finally:
        async with asyncio.timeout(10):
            await database.aclose()


def main(argv: list[str] | None = None) -> int:
    """Return a safe structured error for configuration or incomplete operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    try:
        return asyncio.run(
            run(ConsistencyProcessSettings.load(), args.corpus if args.fix else None)
        )
    except ValidationError:
        code = "CONSISTENCY_CONFIGURATION_INVALID"
    except InsightPilotError as exc:
        code = exc.code
    except TimeoutError:
        code = "CONSISTENCY_DEADLINE_EXCEEDED"
    print(ConsistencyFailure(code=code).model_dump_json(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
