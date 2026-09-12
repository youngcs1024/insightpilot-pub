"""Initialize/validate one collection without deleting or replacing existing data."""

import argparse
import asyncio
import sys

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import InsightPilotError
from app.retrieval.config import MilvusSettings, RetrievalSettings
from app.retrieval.milvus_repo import MilvusRepository


class MilvusOperatorSettings(BaseSettings):
    """Read only retrieval environment fields; never load an API/operator dotenv file."""

    model_config = SettingsConfigDict(
        env_prefix="IP_",
        env_nested_delimiter="__",
        extra="forbid",
        hide_input_in_errors=True,
    )
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


async def initialize(settings: MilvusSettings) -> int:
    """Report the persisted schema and indexes after a successful guard and load."""
    async with MilvusRepository(settings) as repository:
        report = await repository.ensure_collection()
        print(report.model_dump_json(indent=2))
    return 0


def main() -> int:
    """A new --collection target preserves the old index; no drop/reset option exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", help="New collection name for non-destructive recovery")
    args = parser.parse_args()
    try:
        settings = MilvusOperatorSettings().retrieval.milvus
        if args.collection:
            settings = MilvusSettings.model_validate(
                {
                    **settings.model_dump(),
                    "collection": args.collection,
                }
            )
        return asyncio.run(initialize(settings))
    except ValidationError:
        print("Invalid Milvus configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(f"{exc.code}: {exc.user_message}", file=sys.stderr)
        print("Recovery: python -m scripts.milvus_init --collection kb_chunks_v1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
